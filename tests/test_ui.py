from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_forms_demo_and_missing_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path(__file__).resolve().parents[1] / "hydrogen_lab" / "app.py"
    app = AppTest.from_file(str(source), default_timeout=30).run()
    assert not app.exception
    for kind in ["md", "trap", "neb", "tensile"]:
        app.selectbox[0].select(kind).run()
        assert not app.exception
    next(b for b in app.button if b.label == "합성 데이터로 화면 체험").click().run()
    assert not app.exception
    assert any("합성 Brownian" in w.value for w in app.warning)
    assert any(m.label == "수소 확산계수 [m²/s]" and m.value != "판정 보류" for m in app.metric)
    next(b for b in app.button if b.label == "계산 실행").click().run()
    assert any("모델 파일 경로" in e.value for e in app.error)
    assert not app.exception


def test_local_checkpoint_is_selected_automatically(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = tmp_path / 'models'
    models.mkdir()
    # File discovery only; this fixture is never loaded as a real checkpoint.
    checkpoint = models / 'chgnet_0.3.0.pth.tar'
    checkpoint.write_bytes(b'fixture for model path discovery')
    source = Path(__file__).resolve().parents[1] / 'hydrogen_lab' / 'app.py'
    app = AppTest.from_file(str(source), default_timeout=30).run()
    assert not app.exception
    assert app.selectbox[1].value == 'chgnet'
    assert app.text_input(key='checkpoint_chgnet').value == str(checkpoint.resolve())
