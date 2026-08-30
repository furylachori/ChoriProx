"""Fixtures for load testing ChoriProx endpoints.

Load tests use Flask test client with mocked upstream to measure
proxy throughput and latency without external dependencies.
"""
import threading
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="session", autouse=True)
def mock_startup_migrations():
    """Mock startup migrations to prevent Neo4j connection attempts."""
    mock_runner = MagicMock()
    mock_runner.run_pending.return_value = []
    
    with patch('memory.migrations.runner.MigrationRunner', return_value=mock_runner):
        yield mock_runner


@pytest.fixture(scope="session", autouse=True)
def block_openai_client():
    """Block all real OpenAI API calls."""
    mock_chat_response = MagicMock()
    mock_chat_response.choices = [
        MagicMock(
            message=MagicMock(role="assistant", content="mock response"),
            finish_reason="stop",
        )
    ]
    mock_chat_response.model = "gpt-4"
    mock_chat_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )

    mock_completions = MagicMock()
    mock_completions.create.return_value = mock_chat_response

    mock_openai_client = MagicMock()
    mock_openai_client.chat = MagicMock()
    mock_openai_client.chat.completions = mock_completions

    with patch("openai.OpenAI", return_value=mock_openai_client):
        with patch("openai.resources.chat.Completions", return_value=mock_completions):
            yield


@pytest.fixture(scope="session", autouse=True)
def mock_sync_submit():
    """Mock submit_sync to prevent background threads."""
    with patch("memory.sync.submit_sync"):
        yield


@pytest.fixture(autouse=True)
def block_all_http():
    """Block all real HTTP calls with mocked responses."""
    import requests
    from qwen_proxy import _HTTP_SESSION

    default_response = MagicMock()
    default_response.status_code = 200
    default_response.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "mock response"},
                "finish_reason": "stop",
            }
        ]
    }
    default_response.raise_for_status = MagicMock()

    default_get_response = MagicMock()
    default_get_response.status_code = 200
    default_get_response.json.return_value = {}
    default_get_response.raise_for_status = MagicMock()

    def mock_request_side_effect(method, url, *args, **kwargs):
        if method.upper() == "GET":
            return default_get_response
        return default_response

    session_patcher = patch.object(
        _HTTP_SESSION, "request", side_effect=mock_request_side_effect
    )
    session_patcher.start()

    post_patcher = patch("requests.post", return_value=default_response)
    get_patcher = patch("requests.get", return_value=default_get_response)
    post_patcher.start()
    get_patcher.start()

    yield

    session_patcher.stop()
    post_patcher.stop()
    get_patcher.stop()


@pytest.fixture(autouse=True)
def mock_sota_client():
    """Mock SOTAClient to prevent real API calls."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(role="assistant", content="mock sota response"),
            finish_reason="stop",
        )
    ]
    mock_response.model = "sota-model"
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("memory.sota_client.SOTAClient", return_value=mock_client):
        yield mock_client


@pytest.fixture(autouse=True)
def reset_task_graph_singleton():
    """Reset TaskGraph singleton to prevent test pollution."""
    import proxy.context as ctx_module
    original_singleton = ctx_module._task_graph_singleton
    ctx_module._task_graph_singleton = None
    yield
    ctx_module._task_graph_singleton = original_singleton


@pytest.fixture(autouse=True)
def reset_neo4j_circuit_breaker():
    """Reset Neo4j circuit breaker state between tests."""
    import proxy.state as state
    from memory.task_graph import TaskGraph
    
    state.neo4j_circuit_breaker.force_closed()
    TaskGraph._constraint_ensured = False
    
    yield
    
    state.neo4j_circuit_breaker.force_closed()
    TaskGraph._constraint_ensured = False


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiters between tests to prevent test pollution."""
    import proxy.state as state
    
    state.reset_all_rate_limiters()
    
    yield
    
    state.reset_all_rate_limiters()


@pytest.fixture(autouse=True)
def mock_agent_registry_get_or_load():
    """Mock agent_registry.get_or_load to return mock agents."""
    class MockAgent:
        def __init__(self, session_id="test_session"):
            self.active_mode = None
            self._ticket_mode_ts = None
            self._deactivation_callbacks = []
            self._mode_lock = threading.RLock()
            self.session_id = session_id
            self.agent_id = f"agent_{session_id}"
        
        def set_active_mode(self, mode_state, expected_mode=None):
            self.active_mode = mode_state
            return True
        
        def clear_active_mode(self, expected_mode=None):
            self.active_mode = None
            self._ticket_mode_ts = None
            return True
        
        def register_deactivation_callback(self, cb):
            self._deactivation_callbacks.append(cb)
        
        def snapshot_mode(self):
            return self.active_mode
        
        def snapshot_mode_and_ticket_ts(self):
            return self.active_mode, self._ticket_mode_ts
        
        def get_mode_name(self):
            return self.active_mode.name if self.active_mode else None
        
        def is_mode_expired(self):
            return False
    
    agents = {}
    
    def get_or_load_side_effect(session_id):
        if session_id not in agents:
            agents[session_id] = MockAgent(session_id)
        return agents[session_id]
    
    with patch('proxy.agent.agent_registry.registry.get_or_load', side_effect=get_or_load_side_effect):
        yield agents


@pytest.fixture
def client():
    """Create Flask test client with mocked LLM responses for load testing."""
    from qwen_proxy import _HTTP_SESSION, app
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Mock response"},
                "finish_reason": "stop",
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    
    with patch.object(_HTTP_SESSION, "request", return_value=mock_response):
        app.config["TESTING"] = True
        yield app.test_client()
