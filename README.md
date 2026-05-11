# DMS Klebsiella

Instruções para configurar e executar o pipeline de análise computacional.

## Requisitos

- Linux (Ubuntu recomendado)
- Conda instalado ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) é suficiente)
- Licença acadêmica gratuita do PyRosetta (registrar em https://www.pyrosetta.org/downloads antes de instalar)

## Configuração do ambiente (fazer apenas uma vez)

```bash
conda create -n pyrosetta python=3.10
conda activate pyrosetta
pip install -r requirements.txt
pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

## Executar

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pyrosetta
cd /caminho/para/dms_kleb_paper
nohup python scripts/run_dms.py all --no-plot > run.log 2>&1 &
echo $!
```

Substitua `/caminho/para/dms_kleb_paper` pelo caminho real onde a pasta foi descompactada.

O comando roda em segundo plano e registra tudo em `run.log`. O número impresso é o PID do processo.

## Acompanhar o progresso

```bash
tail -f run.log
```

Ou verificar quantas posições de cada proteína já foram processadas:

```bash
watch -n 30 "for p in mgrb phop pmra pmrb phoq; do \
  n=\$(ls \$p/dms_output/csv/ 2>/dev/null | wc -l | tr -d ' '); \
  echo \"\$p: \$n\"; done"
```

## Tempo estimado

2 a 6 dias em um servidor moderno. O processo pode ser interrompido e retomado: basta rodar o mesmo comando novamente e ele continua de onde parou.

## Resultados

Cada proteína gera um arquivo em `<nome>/dms_output/DMS_report.csv`.

Quando todas as proteínas terminarem, gerar o heatmap combinado:

```bash
python scripts/plot_combined.py
```

Os arquivos `combined_dms_heatmap.png` e `combined_dms_heatmap.pdf` serão criados na pasta principal.
