"""Insurance tests for ``_create_db`` polling logic (B-10).

The ``_create_db`` function in ``tests/integration/neo4j_harness.py:142-176``
creates a Neo4j Enterprise database and polls ``SHOW DATABASE`` until it
comes online. A bug in this path (e.g. wrong parameter name, wrong polling
interval, or missing cleanup) would cause the entire ``neo4j_test_db``
fixture to deadlock or leak resources.

These tests pin the invariants:

1. ``CREATE OR REPLACE DATABASE`` is issued against the ``system`` database.
2. After CREATE, the function polls ``SHOW DATABASE $name`` 30 times at
   0.5s intervals (total ~15s budget) with ``{"name": db_name}`` parameters.
3. Returns ``True`` early when ``currentStatus == "online"``.
4. Returns ``False`` (with WARNING) when the 15s budget is exhausted —
   fail-safe so the caller skips the test rather than running migrations
   against a possibly-not-online DB.
5. Returns ``False`` when CREATE itself fails (Community Edition path).
6. Sessions opened for each SHOW DATABASE poll are always closed.
7. The function is idempotent — calling twice on the same name does not
   raise; both calls return ``True`` when the DB comes online (CREATE OR
   REPLACE is no-op if exists).
"""
import unittest
from unittest.mock import MagicMock, patch

from tests.integration.neo4j_harness import _create_db


def _create_result_mock():
    """Build a mock for a CREATE OR REPLACE DATABASE run result.

    ``result.consume()`` must be callable.
    """
    r = MagicMock()
    r.consume = MagicMock()
    return r


def _build_driver(create_ok=True, status_sequence=None, total_create_calls=1):
    """Build a mock driver wired with sequential session behaviour.

    The first call to ``driver.session()`` returns the CREATE session.
    Subsequent calls return SHOW DATABASE sessions that yield the next
    status from ``status_sequence``. Once a SHOW session's ``.close()``
    has been called, the NEXT call to ``driver.session()`` returns
    another CREATE session — this lets tests verify idempotency (two
    ``_create_db`` invocations on the same driver) by passing
    ``total_create_calls=2``.

    Past the end of ``status_sequence``, SHOW sessions return ``None``
    rows (treating missing data as "not online yet").

    Returns ``(driver, create_session, show_sessions)``.
    """
    if status_sequence is None:
        status_sequence = []

    create_session = MagicMock(name="create_session")
    if create_ok:
        create_session.run = MagicMock(return_value=_create_result_mock())
    else:
        create_session.run = MagicMock(side_effect=RuntimeError("UnsupportedAdministrationCommand"))
    create_session.close = MagicMock()

    show_sessions = []
    iter_status = iter(status_sequence)
    creates_issued = [0]
    awaiting_create = [True]

    def session_factory(*args, **kwargs):
        if awaiting_create[0] and creates_issued[0] < total_create_calls:
            creates_issued[0] += 1
            awaiting_create[0] = False
            return create_session
        s = MagicMock(name=f"show_session_{len(show_sessions)}")
        try:
            row = next(iter_status)
        except StopIteration:
            row = None
        run_result = MagicMock()
        run_result.single = MagicMock(return_value=row)
        s.run = MagicMock(return_value=run_result)
        original_close = s.close

        def close_and_arm(*cargs, **ckwargs):
            result = original_close(*cargs, **ckwargs)
            awaiting_create[0] = True
            return result

        s.close = MagicMock(side_effect=close_and_arm)
        s._database_arg = kwargs.get("database")
        show_sessions.append(s)
        return s

    driver = MagicMock()
    driver.session = MagicMock(side_effect=session_factory)
    return driver, create_session, show_sessions


def _query_text(query_or_query_obj):
    """Extract the raw text from a Query wrapper or pass-through string."""
    return getattr(query_or_query_obj, "text", None) or str(query_or_query_obj)


class TestCreateDbCreatePhase(unittest.TestCase):
    """Tests for the CREATE OR REPLACE DATABASE call."""

    def test_create_uses_system_database(self):
        """CREATE is issued against the ``system`` database (not the target)."""
        driver, create_session, show_sessions = _build_driver(
            status_sequence=[{"currentStatus": "online"}]
        )

        with patch("time.sleep"):
            _create_db(driver, "test_db")

        self.assertIsNotNone(create_session.run.call_args)
        for s in show_sessions:
            self.assertEqual(s._database_arg, "system")

    def test_create_query_includes_db_name(self):
        """CREATE OR REPLACE DATABASE statement interpolates db_name."""
        driver, create_session, _ = _build_driver(status_sequence=[{"currentStatus": "online"}])

        with patch("time.sleep"):
            _create_db(driver, "my_unique_db")

        create_call = create_session.run.call_args
        self.assertIsNotNone(create_call)
        query = _query_text(create_call.args[0])
        self.assertIn("CREATE OR REPLACE DATABASE", query)
        self.assertIn("my_unique_db", query)

    def test_create_failure_returns_false(self):
        """If CREATE raises (Community Edition), function returns False without polling."""
        driver, create_session, show_sessions = _build_driver(
            create_ok=False, status_sequence=[]
        )

        result = _create_db(driver, "any_db")
        self.assertFalse(result)
        create_session.close.assert_called_once()
        self.assertEqual(len(show_sessions), 0)

    def test_create_session_closed_on_failure(self):
        """Session is closed even when CREATE raises."""
        driver, create_session, _ = _build_driver(create_ok=False, status_sequence=[])

        _create_db(driver, "any_db")
        create_session.close.assert_called_once()

    def test_create_session_closed_on_success(self):
        """Session is closed after successful CREATE."""
        driver, create_session, _ = _build_driver(status_sequence=[{"currentStatus": "online"}])

        with patch("time.sleep"):
            _create_db(driver, "test_db")

        create_session.close.assert_called_once()


class TestCreateDbPollingLoop(unittest.TestCase):
    """Tests for the SHOW DATABASE polling loop."""

    def test_returns_true_when_first_poll_reports_online(self):
        """DB online on first poll → returns True without further polling."""
        driver, _, _ = _build_driver(status_sequence=[{"currentStatus": "online"}])

        with patch("time.sleep") as mock_sleep:
            result = _create_db(driver, "test_db")

        self.assertTrue(result)
        mock_sleep.assert_called_once_with(0.5)

    def test_returns_true_when_later_poll_reports_online(self):
        """DB online after N polls → returns True early (does not exhaust loop)."""
        driver, _, _ = _build_driver(
            status_sequence=[
                {"currentStatus": "starting"},
                {"currentStatus": "starting"},
                {"currentStatus": "online"},
            ]
        )

        with patch("time.sleep") as mock_sleep:
            result = _create_db(driver, "test_db")

        self.assertTrue(result)
        self.assertEqual(mock_sleep.call_count, 3)

    def test_polls_30_times_on_timeout(self):
        """When DB never reports online, polls exactly 30 times then warns."""
        driver, _, _ = _build_driver(
            status_sequence=[{"currentStatus": "starting"}] * 30
        )

        with patch("time.sleep") as mock_sleep:
            with self.assertLogs("tests.integration.neo4j_harness", level="WARNING") as cm:
                result = _create_db(driver, "test_db")

        self.assertFalse(result)
        self.assertEqual(mock_sleep.call_count, 30)
        self.assertTrue(
            any("did not come online within 15s" in msg for msg in cm.output),
            f"expected timeout warning; got {cm.output}",
        )

    def test_timeout_returns_false_not_true(self):
        """M4: on 15s timeout the function returns False (fail-safe skip path),
        NOT True. Returning True would let the caller run migrations on a
        possibly-not-online DB → flaky failures.
        """
        driver, _, _ = _build_driver(
            status_sequence=[{"currentStatus": "starting"}] * 30
        )

        with patch("time.sleep"):
            result = _create_db(driver, "timeout_db")

        self.assertFalse(result)

    def test_timeout_warning_indicates_skip(self):
        """M4: the timeout WARNING logs that the DB will be skipped."""
        driver, _, _ = _build_driver(
            status_sequence=[{"currentStatus": "starting"}] * 30
        )

        with patch("time.sleep"):
            with self.assertLogs("tests.integration.neo4j_harness", level="WARNING") as cm:
                _create_db(driver, "timeout_db")

        self.assertTrue(
            any("returning False" in msg or "skipped" in msg for msg in cm.output),
            f"expected skip-indicating warning; got {cm.output}",
        )

    def test_polling_interval_is_half_second(self):
        """Each poll iteration sleeps for 0.5s."""
        driver, _, _ = _build_driver(
            status_sequence=[{"currentStatus": "starting"}] * 5
            + [{"currentStatus": "online"}]
        )

        with patch("time.sleep") as mock_sleep:
            _create_db(driver, "test_db")

        for call in mock_sleep.call_args_list:
            self.assertEqual(call.args, (0.5,))

    def test_show_database_opens_session_against_system(self):
        """Each SHOW DATABASE poll opens a session against the ``system`` DB."""
        driver, _, show_sessions = _build_driver(
            status_sequence=[{"currentStatus": "starting"}] * 3
            + [{"currentStatus": "online"}]
        )

        with patch("time.sleep"):
            _create_db(driver, "alpha_db")

        self.assertEqual(len(show_sessions), 4)
        for s in show_sessions:
            self.assertEqual(s._database_arg, "system")

    def test_show_run_uses_correct_query_and_params(self):
        """The SHOW DATABASE statement includes the ``$name`` parameter placeholder and the name."""
        driver, _, show_sessions = _build_driver(
            status_sequence=[{"currentStatus": "online"}]
        )

        with patch("time.sleep"):
            _create_db(driver, "beta_db")

        self.assertEqual(len(show_sessions), 1)
        show_run = show_sessions[0].run.call_args
        self.assertIsNotNone(show_run)
        query, params = show_run.args
        self.assertIn("SHOW DATABASE", _query_text(query))
        self.assertIn("$name", _query_text(query))
        self.assertEqual(params, {"name": "beta_db"})

    def test_each_poll_opens_own_session_and_closes_it(self):
        """Every poll iteration opens a session and closes it (no leak)."""
        driver, _, show_sessions = _build_driver(
            status_sequence=[{"currentStatus": "online"}] * 5
        )

        with patch("time.sleep"):
            _create_db(driver, "test_db")

        self.assertEqual(len(show_sessions), 1)
        for s in show_sessions:
            s.close.assert_called_once()

    def test_poll_with_null_row_does_not_return_true(self):
        """If ``.single()`` returns ``None``, keep polling (do not treat as online)."""
        driver, _, _ = _build_driver(
            status_sequence=[None, None, {"currentStatus": "online"}]
        )

        with patch("time.sleep") as mock_sleep:
            result = _create_db(driver, "test_db")

        self.assertTrue(result)
        self.assertEqual(mock_sleep.call_count, 3)

    def test_non_online_status_keeps_polling(self):
        """Any status other than ``online`` keeps the loop running."""
        driver, _, _ = _build_driver(
            status_sequence=[
                {"currentStatus": "starting"},
                {"currentStatus": "store copying"},
                {"currentStatus": "online"},
            ]
        )

        with patch("time.sleep") as mock_sleep:
            result = _create_db(driver, "test_db")

        self.assertTrue(result)
        self.assertEqual(mock_sleep.call_count, 3)

    def test_missing_current_status_key_keeps_polling(self):
        """A row with no ``currentStatus`` key (e.g. error response) keeps polling."""
        driver, _, _ = _build_driver(
            status_sequence=[{"someOtherKey": "x"}, {"currentStatus": "online"}]
        )

        with patch("time.sleep") as mock_sleep:
            result = _create_db(driver, "test_db")

        self.assertTrue(result)
        self.assertEqual(mock_sleep.call_count, 2)

    def test_sessions_closed_even_on_timeout(self):
        """Even when the 15s budget is exhausted, all polling sessions are closed."""
        driver, _, show_sessions = _build_driver(
            status_sequence=[{"currentStatus": "starting"}] * 30
        )

        with patch("time.sleep"):
            _create_db(driver, "test_db")

        self.assertEqual(len(show_sessions), 30)
        for s in show_sessions:
            s.close.assert_called_once()


class TestCreateDbIdempotency(unittest.TestCase):
    """Calling ``_create_db`` twice on the same name must be safe."""

    def test_calling_twice_returns_true_both_times(self):
        """``CREATE OR REPLACE DATABASE`` is no-op on second call; both succeed."""
        driver, create_session, _ = _build_driver(
            status_sequence=[{"currentStatus": "online"}, {"currentStatus": "online"}],
            total_create_calls=2,
        )

        with patch("time.sleep"):
            r1 = _create_db(driver, "same_db")
            r2 = _create_db(driver, "same_db")

        self.assertTrue(r1)
        self.assertTrue(r2)
        self.assertEqual(create_session.run.call_count, 2)


class TestCreateDbLogging(unittest.TestCase):
    """Logging behaviour is part of the contract — pin it."""

    def test_create_failure_logs_at_debug(self):
        """CREATE failure (Community Edition) logs at DEBUG, not WARNING."""
        driver, _, _ = _build_driver(create_ok=False, status_sequence=[])

        with self.assertLogs("tests.integration.neo4j_harness", level="DEBUG") as cm:
            _create_db(driver, "any_db")

        debug_msgs = [m for m in cm.output if "CREATE DATABASE failed" in m]
        self.assertEqual(len(debug_msgs), 1)
        self.assertTrue(
            all("UnsupportedAdministrationCommand" in m for m in debug_msgs),
            f"expected exception details in debug log; got {cm.output}",
        )

    def test_timeout_logs_warning_with_db_name(self):
        """Timeout warning includes the db name so operators can find the issue."""
        driver, _, _ = _build_driver(
            status_sequence=[{"currentStatus": "starting"}] * 30
        )

        with patch("time.sleep"):
            with self.assertLogs("tests.integration.neo4j_harness", level="WARNING") as cm:
                _create_db(driver, "slow_db")

        warning_msgs = [m for m in cm.output if "did not come online within 15s" in m]
        self.assertEqual(len(warning_msgs), 1)
        self.assertIn("slow_db", warning_msgs[0])
