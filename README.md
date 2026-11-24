# Hydrogen Lab — Fe·철강 수소취성 연구용 MLIP / MD 도구

DFT를 실행하지 않고 **MACE 또는 CHGNet + ASE**로 새 구조의 수소 이동과 재료 응답을 계산하는 로컬 소프트웨어입니다. 한국어 화면과 CLI를 제공합니다. Windows PC를 기준으로 만들었으며 Python 엔진은 다른 운영체제에서도 사용할 수 있습니다.

이 소스는 [hydrogen-local-energetics](https://github.com/lyullee/hydrogen-local-energetics)를 검토해 만든 별도 0.1 버전입니다. 원본 논문 재현 자료를 수정하거나 원본 GitHub에 업로드하지 않았습니다. 원본의 모델 식별자·해시를 참조하며 DFT 데이터·의사퍼텐셜·모델 가중치를 배포물에 포함하지 않습니다. 원본의 유실된 CHGNet NEB 생성 코드를 복구했다고 주장하지 않습니다.

## 기능

| 분석 | 입력 | 결과 |
|---|---|---|
| 수소 확산 MD | 모재+H 구조, 온도, 시간, 시드 | 고정 셀 NVT 궤적, 모재 이동 보정 MSD, 정상확산 점검, D [m²/s] |
| 결함 포획 | 벌크/벌크+H/결함/결함+H | 포획에너지 [eV], 양수이면 벌크 기준 자리보다 포획이 유리 |
| 수소 이동 NEB | 원자 순서가 동일한 시작/끝 구조 | CI-NEB 정·역방향 이동 장벽 [eV] |
| 수소 유무 인장 비교 | 수소 없음/포함 구조 | 0 K 준정적 응력–변형률 곡선, 두 곡선에서 최대점 확인 시 최대응력 감소율 |

계산 조건, 구조·모델 SHA-256, 패키지 버전, 로그, 궤적과 결과 JSON을 실행별 폴더에 저장합니다. 미수렴·모델 불일치·유효하지 않은 입력은 실패로 기록합니다. 계산된 지표를 하나의 수소취성 등급으로 환산하지 않습니다.

## Windows에서 실행

Python 3.11 이상이 필요합니다. 처음에는 Python 3.12와 CPU CHGNet 조합을 권합니다. PowerShell에서 **이 README가 있는 폴더**로 이동한 뒤 실행합니다.

**간편 실행:** ZIP을 풀고 `START.cmd`를 더블클릭하세요. 처음 실행할 때 필요한 패키지와 기본 CHGNet 모델을 준비하고, 서버가 준비되면 `http://127.0.0.1:8511`을 엽니다. 이미 실행 중이면 같은 서버의 화면을 다시 엽니다. 화면에서 CHGNet과 로컬 모델 경로가 자동 선택됩니다. 첫 설치에는 인터넷이 필요하며 수 분 걸릴 수 있습니다. 시작 로그는 `.runtime/`에 저장됩니다.

```powershell
# 설치 + 원본 저장소에 기록된 CHGNet 모델 다운로드 및 해시 검증
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Backend chgnet -DownloadCheckpoint

# 한국어 화면 시작 (127.0.0.1에만 연결)
powershell -ExecutionPolicy Bypass -File .\launch.ps1
```

화면에서 계산기 `CHGNet`, 설치된 모델 경로, 장치 `cpu`, 지원 원소 `Fe,H`가 기본으로 선택됩니다. `Fe–H 모델 연결 확인`으로 실제 에너지·힘·응력 계산을 점검한 뒤, bcc Fe 예제와 원하는 분석을 선택해 실행합니다. GPU를 쓸 경우 해당 장치용 PyTorch 설치가 추가로 필요합니다. 54 Fe+1 H 기본 예제도 CPU에서는 시간이 걸릴 수 있으며 MD·NEB·포획·인장 계산 진행 상태를 화면에 표시합니다.

모델 없이 화면과 궤적 후처리부터 확인할 경우:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Backend none
powershell -ExecutionPolicy Bypass -File .\launch.ps1
```

`MD 궤적 분석 → 합성 데이터로 화면 체험`은 인공 Brownian 궤적을 사용합니다. **합성 데모는 Fe 물성이나 MLIP 계산 결과가 아닙니다.**

MACE 사용자는 `setup.ps1 -Backend mace -DownloadCheckpoint`로 설치하거나 신뢰하는 로컬 `.model` 파일을 지정합니다. MACE-OMat-0의 Academic Software License 등 모델별 이용 조건은 upstream을 확인해야 합니다. 학습 모델 파일은 신뢰하는 출처의 것만 여세요.

## CLI

```powershell
.venv/Scripts/python.exe -m hydrogen_lab.cli examples --out examples
# examples/*.json의 model.backend, path, device를 실제 모델에 맞게 수정
.venv/Scripts/python.exe -m hydrogen_lab.cli run examples/md.json --out runs
.venv/Scripts/python.exe -m hydrogen_lab.cli run examples/trap.json --out runs
.venv/Scripts/python.exe -m hydrogen_lab.cli run examples/neb.json --out runs
.venv/Scripts/python.exe -m hydrogen_lab.cli run examples/tensile.json --out runs
.venv/Scripts/python.exe -m hydrogen_lab.cli analyze examples/synthetic_diffusion.npz --out demo_report.json
```

구조·모델의 상대 경로는 **설정 JSON이 있는 폴더**를 기준으로 해석합니다. 기본 JSON은 CHGNet 경로를 사용합니다. MACE이면 `"backend": "mace"`, `"path": "../models/mace-omat-0-medium.model"`로 바꾸세요. `model.sha256`을 추가하면 예상 모델과 일치하는지 실행 전에 검사합니다.

기본 MD는 600 K, 0.5 fs, 평형화 2,000 단계, 생산 20,000 단계(10 ps), H 1개입니다. 이는 계산 흐름을 시작하기 위한 값이며 수렴된 물성을 보장하지 않습니다. 장시간 실행은 `max_lag`도 늘려 전체 생산시간의 20–50% 적합 구간을 포함해야 합니다.

## 결과 해석과 현재 범위

- 사전학습 MLIP는 DFT로 학습되었을 수 있습니다. 여기서 DFT 제외는 **새 DFT 계산을 실행하지 않는다**는 뜻입니다.
- 원소 지원만으로 Fe–H, 공공, 입계, 전위, 큰 변형, 파단 부근의 정확도가 입증되지는 않습니다. 검증 메모는 사용자 기록이며 자동 검증 인증이 아닙니다.
- Fe 예제는 bcc 단결정과 공공입니다. 실제 철강 조성·다결정·열처리 조직을 자동 재현하지 않습니다. 합금·결함 구조는 파일로 입력하고 해당 원소와 환경에 대해 모델을 검증해야 합니다.
- 포획·NEB·인장은 0 K 고정 셀 계산이며, 인장은 횡방향 치수 고정 준정적 원자 이완입니다. 유한온도 인장 MD나 균열 진전 계산이 아닙니다.
- MD 확산은 고전 핵 근사입니다. 수소 양자 핵 효과·영점에너지·터널링과 실험 시간척도 보정은 포함하지 않습니다.
- 수소취성 등급, 연신율, K_IC, 설비 수명, HELP/HEDE 기구의 확정 판정은 제공하지 않습니다. 실험과 모델 적용 범위·시간·셀 크기·수소농도·변형률·여러 시드에 대한 검증이 다음 단계입니다.

수식·부호·입력 규약·출처는 [docs/METHODS.md](docs/METHODS.md), 실제 검증 범위는 [docs/VALIDATION.md](docs/VALIDATION.md)를 확인하세요.

## 개발 검증

```powershell
.venv/Scripts/python.exe -m pip install -e ".[test]"
.venv/Scripts/python.exe -m pytest -q
```

소스 라이선스는 MIT입니다. 외부 모델 라이선스는 각 upstream을 따릅니다.
