from mirage.cli import main
from mirage.dashboard import create_app


def test_cli_help(capsys) -> None:
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    assert "mirage" in output


def test_dashboard_once_without_run(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    code = main(["dashboard", "--once"])
    assert code == 0
    assert "Mirage" in capsys.readouterr().out


def test_dashboard_app_can_be_created(tmp_path) -> None:
    app = create_app(tmp_path)
    routes = {route.path for route in app.routes}
    assert "/api/state" in routes
    assert "/artifacts/{artifact_path:path}" in routes


def test_dashboard_artifact_paths_stay_inside_run_dir(tmp_path) -> None:
    app = create_app(tmp_path)
    artifact_route = next(route for route in app.routes if route.path == "/artifacts/{artifact_path:path}")

    response = artifact_route.endpoint("../outside.txt")

    assert response.status_code == 403
