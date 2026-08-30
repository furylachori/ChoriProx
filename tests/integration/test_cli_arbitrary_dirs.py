"""US-046: Clients are CLI Agents Running from Arbitrary Directories.

The proxy must accept requests regardless of the agent's working directory
and must NOT restrict requests based on file paths or working directory.
"""

import pytest


@pytest.mark.integration
class TestUS046ArbitraryWorkingDirectories:
    """US-046: proxy accepts requests with any working_directory/cwd field."""

    def test_request_with_working_directory_field_accepted(self, proxy_with_mock):
        """A request carrying a working_directory field must be accepted."""
        client, handler, _ = proxy_with_mock

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi",
                "messages": [{"role": "user", "content": "hello"}],
                "working_directory": "/some/arbitrary/path/project",
            },
            headers={"Authorization": "Bearer test-token"},
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert "choices" in data

    def test_request_with_cwd_field_accepted(self, proxy_with_mock):
        """A request carrying a cwd field must be accepted."""
        client, handler, _ = proxy_with_mock

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi",
                "messages": [{"role": "user", "content": "hello"}],
                "cwd": "/Users/dev/projects/my-app",
            },
            headers={"Authorization": "Bearer test-token"},
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert "choices" in data

    @pytest.mark.parametrize("path_value", [
        "/tmp",
        "/Users/someone/Desktop/code",
        "C:\\Users\\dev\\projects",
        "/var/lib/data",
        ".",
        "..",
        "~/projects/my-app",
        "/path with spaces/project",
        "/path/with/special/chars!@#$%",
        "",
    ])
    def test_various_path_values_accepted(self, proxy_with_mock, path_value):
        """Various path formats in working_directory must all be accepted."""
        client, handler, _ = proxy_with_mock

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi",
                "messages": [{"role": "user", "content": "hello"}],
                "working_directory": path_value,
            },
            headers={"Authorization": "Bearer test-token"},
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert "choices" in data
        assert "error" not in data

    def test_request_with_file_paths_in_messages_accepted(self, proxy_with_mock):
        """File paths embedded in message content must not be rejected."""
        client, handler, _ = proxy_with_mock

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi",
                "messages": [
                    {
                        "role": "user",
                        "content": "Look at /etc/passwd and C:\\Windows\\System32\\config",
                    }
                ],
            },
            headers={"Authorization": "Bearer test-token"},
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert "choices" in data
