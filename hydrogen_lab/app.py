"""Korean local Streamlit interface. Start with `hydrogen-lab ui`."""
import io
import importlib.util
import json
from pathlib import Path
from uuid import uuid4
import pandas as pd
import streamlit as st
from hydrogen_lab.cli import examples
from hydrogen_lab.runner import run, analyze_npz
from hydrogen_lab.workflows import WORKFLOWS
from hydrogen_lab.models import load_calculator
from hydrogen_lab.registry import REGISTRY
from hydrogen_lab.structures import iron_examples
from hydrogen_lab import __version__

st.set_page_config(page_title="Hydrogen Lab · MLIP / MD", page_icon="⚛️", layout="wide")
st.markdown("""<style>
.block-container {max-width:1160px;padding-top:2.4rem}
[data-testid="stMetric"] {background:#edf5f5;border:1px solid #d2e3e2;border-radius:12px;padding:16px}
h1 {letter-spacing:-1px} .stCaption {color:#516575}
</style>""", unsafe_allow_html=True)
st.caption(f"HYDROGEN LAB  /  Fe · STEEL  /  RESEARCH EDITION {__version__}")
st.title("수소의 이동에서 재료 응답까지")
st.write("MACE·CHGNet과 분자동역학으로 수소취성 관련 지표를 비교합니다. DFT 계산은 실행하지 않습니다.")
st.info("연구용 1차 버전입니다. 모델의 Fe–H·결함·변형 환경 정확도와 계산 수렴을 확인해야 합니다. 결과는 실험 수소취성 등급이나 설비 수명 판정이 아닙니다.")
root = Path.cwd()
example_root = root / "examples"
if not (example_root / "md.json").exists():
    examples(example_root)


def show_result(report, key_prefix="current"):
    result = report.get("result", report)
    cols = st.columns(3)
    cols[0].metric("분석 상태", "완료" if report.get("status") == "completed" else "궤적 후처리")
    if "D_m2_s" in result:
        cols[1].metric("수소 확산계수 [m²/s]", f"{result['D_m2_s']:.3e}" if result["D_m2_s"] is not None else "판정 보류")
        cols[2].metric("MSD 적합 R²", f"{result['fit_r2']:.4f}" if result["fit_r2"] is not None else "—")
        st.line_chart(pd.DataFrame({"지연 시간 [fs]": result["lag_fs"], "H MSD [Å²]": result["msd_A2"]}).set_index("지연 시간 [fs]"))
        st.caption(f"모재 질량중심 이동 보정 · 다중 시간원점 · 적합 구간 {result['fit_start_fs']:.1f}–{result['fit_end_fs']:.1f} fs")
    elif "trapping_energy_eV" in result:
        cols[1].metric("수소 포획에너지 [eV]", f"{result['trapping_energy_eV']:.4f}")
        st.caption("양수: 입력한 벌크 수소 자리보다 결함 자리가 에너지상 유리함")
        st.dataframe(pd.DataFrame(result["energies_eV"].items(), columns=["구조", "총에너지 [eV]"]), hide_index=True)
    elif "forward_barrier_eV" in result:
        cols[1].metric("정방향 장벽 [eV]", f"{result['forward_barrier_eV']:.4f}")
        cols[2].metric("역방향 장벽 [eV]", f"{result['reverse_barrier_eV']:.4f}")
        st.line_chart(pd.DataFrame({"상대 에너지 [eV]": result["relative_energies_eV"]}).rename_axis("경로 이미지"))
    elif "curves" in result:
        loss = result["peak_strength_reduction_fraction"]
        cols[1].metric("최대 인장응력 감소율", f"{100 * loss:.1f}%" if loss is not None else "최대점 미확인")
        data = {"변형률": [r[0] for r in result["curves"]["clean"]],
                "수소 없음 [GPa]": [r[1] for r in result["curves"]["clean"]],
                "수소 포함 [GPa]": [r[1] for r in result["curves"]["hydrogenated"]]}
        st.line_chart(pd.DataFrame(data).set_index("변형률"))
        st.caption("0 K 준정적 인장 · 횡방향 셀 고정 · 유한온도 인장 MD와 구분")
    with st.expander("해석 조건·수렴 확인", expanded=True):
        for warning in result.get("warnings", []):
            st.write("• " + warning)
    if report.get("run_directory"):
        st.code(report["run_directory"], language=None)
    st.download_button("결과 JSON 저장", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                       file_name="hydrogen_report.json", mime="application/json", key=key_prefix + "_download_" + str(report.get("run_directory", "analysis")))


new, analysis, history = st.tabs(["새 MLIP 계산", "MD 궤적 분석", "실행 기록·방법"])
with new:
    labels = {"md": "MD · 수소 확산", "trap": "결함 · 수소 포획에너지", "neb": "NEB · 수소 이동 장벽", "tensile": "준정적 인장 · 수소 유무 비교"}
    kind = st.selectbox("분석 종류", list(labels), format_func=labels.get)
    default = json.loads((example_root / f"{kind}.json").read_text(encoding="utf-8"))
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("1. MLIP 모델")
        backend = st.selectbox("계산기", ["chgnet", "mace"], format_func=lambda x: "MACE" if x == "mace" else "CHGNet")
        local_model = root / "models" / REGISTRY[backend]["filename"]
        checkpoint = st.text_input("로컬 모델 파일 경로", value=str(local_model.resolve()) if local_model.is_file() else "",
                                   placeholder="설치된 모델이 있으면 자동으로 선택됩니다.", key=f"checkpoint_{backend}")
        device = st.selectbox("계산 장치", ["cpu", "cuda", "mps"])
        elements = st.text_input("모델이 지원하는 원소 (쉼표 구분)", "Fe,H")
        note = st.text_area("모델 검증 메모", placeholder="Fe–H, 공공, 변형 구조 검증 자료·오차·적용 범위")
        if checkpoint and Path(checkpoint).is_file() and importlib.util.find_spec(backend):
            st.success("계산 준비 완료 · 로컬 모델이 연결되어 있습니다.")
        else:
            st.warning("선택한 모델이 준비되지 않았습니다. START.cmd로 실행하면 기본 CHGNet 설치와 모델 준비를 진행합니다.")
        st.caption("기본 CHGNet은 설치 시 준비합니다. 지원 원소는 사용자 선언이며 정확도 검증을 대신하지 않습니다.")
        if st.button("Fe–H 모델 연결 확인"):
            try:
                import numpy as np
                spec = {"backend": backend, "path": checkpoint.strip(), "device": device,
                        "supported_elements": [s.strip() for s in elements.split(",") if s.strip()], "validation_note": note}
                with st.spinner("Fe16H 구조의 실제 MLIP 에너지·힘·응력을 확인하는 중입니다."):
                    a = iron_examples(2)["bulk_h"]
                    a.calc, provenance = load_calculator(spec, a.get_chemical_symbols())
                    energy = float(a.get_potential_energy())
                    forces, stress = a.get_forces(), a.get_stress()
                    if not np.isfinite(energy) or not np.isfinite(forces).all() or not np.isfinite(stress).all():
                        raise RuntimeError("모델이 유효한 에너지·힘·응력을 반환하지 않았습니다.")
                    st.session_state["model_check"] = {"model": provenance, "formula": a.get_chemical_formula(),
                                                       "energy_eV": energy, "max_abs_force_eV_A": float(np.abs(forces).max()),
                                                       "stress_eV_A3": stress.tolist()}
                st.success("모델 연결 성공 · 실제 Fe–H 에너지·힘·응력 계산을 확인했습니다.")
                st.json(st.session_state["model_check"], expanded=False)
            except Exception as error:
                st.error(f"모델 연결 확인 실패: {error}")
    with right:
        st.subheader("2. 구조와 계산 조건")
        source = st.radio("구조 입력", ["bcc Fe 예제", "파일 업로드"], horizontal=True)
        uploads = {}
        if source == "파일 업로드":
            for key in WORKFLOWS[kind][1]:
                uploads[key] = st.file_uploader(key, type=["extxyz", "xyz", "cif", "vasp", "traj"], key=f"{kind}_{key}")
            st.caption("3차원 주기 셀 필요 · NEB 원자 순서 일치 · 인장 x/y/z 축에 정렬된 직교 셀")
        else:
            st.write("54 Fe 원자 bcc 셀 · 수소 포함 구조는 H 1개 · 결함 예제는 공공 1개")
            st.caption("예제 격자상수 2.86 Å는 시작값입니다. 모델·온도에 맞는 셀을 검증하세요.")
        options_text = st.text_area("계산 조건 (JSON)", json.dumps(default["options"], indent=2), height=210, key=f"options_{kind}")
        st.caption("MD 기본값: 600 K · 0.5 fs · 평형화 1 ps + 생산 10 ps. 연구 결과에는 시간·크기·시드 수렴이 필요합니다.")
    if st.button("계산 실행", type="primary"):
        try:
            if not checkpoint.strip():
                raise ValueError("로컬 MLIP 모델 파일 경로를 입력하세요.")
            structures = {k: str((example_root / v).resolve()) for k, v in default["structures"].items()}
            if source == "파일 업로드":
                if any(v is None for v in uploads.values()):
                    raise ValueError("필요한 구조 파일을 모두 넣어 주세요.")
                upload_root = root / "runs" / "inputs" / uuid4().hex
                upload_root.mkdir(parents=True)
                for key, upload in uploads.items():
                    path = upload_root / (key + Path(upload.name).suffix)
                    path.write_bytes(upload.getvalue())
                    structures[key] = str(path.resolve())
            config = {"workflow": kind, "structures": structures, "options": json.loads(options_text),
                      "model": {"backend": backend, "path": checkpoint.strip(), "device": device,
                                "supported_elements": [s.strip() for s in elements.split(",") if s.strip()], "validation_note": note}}
            with st.spinner("계산 중입니다. 계산 단계·궤적·로그는 runs 폴더에 저장됩니다."):
                bar = st.progress(0.0, text="계산을 준비하는 중입니다.")
                st.session_state["mlip_report"] = run(config, root, root / "runs",
                                                       progress=lambda fraction, message: bar.progress(fraction, text=message))
            st.success("계산과 결과 저장을 마쳤습니다.")
        except Exception as error:
            st.error(f"{type(error).__name__}: {error}")
            st.caption("실행을 시작한 경우 runs 하위 report.json에 실패 원인과 입력 기록이 남습니다.")
    if "mlip_report" in st.session_state:
        show_result(st.session_state["mlip_report"])

with analysis:
    st.subheader("궤적에서 수소 확산계수 계산")
    st.write("이 프로그램이 생성한 unwrapped.npz를 읽습니다. 모델 없이도 후처리할 수 있습니다.")
    st.caption("필수 배열: positions_A, time_fs, symbols, masses_amu, unwrapped=True. 생산 구간만 사용하며 셀은 고정이어야 합니다.")
    upload = st.file_uploader("연속 좌표 NPZ", type="npz")
    max_lag = st.number_input("최대 지연 프레임 수", 10, 100000, 1000)
    c1, c2 = st.columns(2)
    if c1.button("업로드 궤적 분석"):
        try:
            if upload is None:
                raise ValueError("NPZ 궤적을 넣어 주세요.")
            result = analyze_npz(io.BytesIO(upload.getvalue()), max_lag=int(max_lag))
            st.session_state["trajectory_report"] = {"source": upload.name, "synthetic": result.get("synthetic"), "result": result}
        except Exception as error:
            st.error(str(error))
    if c2.button("합성 데이터로 화면 체험"):
        result = analyze_npz(example_root / "synthetic_diffusion.npz")
        st.session_state["trajectory_report"] = {"synthetic": True, "not_a_material_prediction": True, "result": result}
    if "trajectory_report" in st.session_state:
        report = st.session_state["trajectory_report"]
        if report.get("synthetic"):
            st.warning("합성 Brownian 궤적 데모입니다. Fe·철강의 MLIP 계산값이나 물성 예측이 아닙니다.")
        show_result(report)

with history:
    st.subheader("계산 기록")
    paths = sorted((root / "runs").glob("*/report.json"), reverse=True) if (root / "runs").exists() else []
    if paths:
        selected = st.selectbox("실행 선택", paths, format_func=lambda p: p.parent.name)
        report = json.loads(selected.read_text(encoding="utf-8"))
        st.json(report, expanded=False)
        if report.get("status") == "completed":
            show_result(report, key_prefix="history")
    else:
        st.write("저장된 MLIP 계산이 없습니다.")
    st.subheader("분석 범위")
    st.write("확산: 모재 질량중심 보정 + 다중 시간원점 MSD + 3차원 Einstein 관계. 포획: 동일 셀에서 수소 삽입 에너지 차이. 이동 장벽: 끝점 이완 후 climbing-image NEB. 인장: 0 K에서 원자 위치를 이완하는 준정적 비교.")
    st.write("DFT 엔진과 의사퍼텐셜은 필요하지 않습니다. 사전학습 MLIP의 학습 자료에는 DFT가 포함될 수 있습니다.")
    st.write("다결정 생성, 균열 진전, 유한온도 인장 MD, HELP/HEDE 구분, 양자 핵 효과, 실험 기반 수소취성 등급은 이번 버전의 범위에 포함되지 않습니다.")
