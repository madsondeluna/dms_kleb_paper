# Deep Mutational Scanning of MgrB, PhoP, PhoQ, PmrA, and PmrB in *Klebsiella pneumoniae* under antimicrobial resistance phenotypes

Instruções para configurar e executar o pipeline de análise computacional em um servidor Linux.

## 1. Instalar o Miniconda (se ainda não tiver)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Siga as instruções na tela. Ao final, feche e reabra o terminal (ou rode `source ~/.bashrc`).

Verifique se funcionou:

```bash
conda --version
```

## 2. Baixar o repositório

```bash
git clone https://github.com/madsondeluna/dms_kleb_paper.git
cd dms_kleb_paper
```

## 3. PyRosetta (sem cadastro)

O `pyrosetta-installer` baixa os wheels oficiais sem credenciais. A instalação é feita no próximo passo. Uso comercial requer licença separada (license@uw.edu).

## 4. Configurar o ambiente (fazer apenas uma vez)

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda create -n pyrosetta python=3.10 -y
conda activate pyrosetta
pip install -r requirements.txt
pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
```

## 5. Executar o pipeline

### Mac ou Linux (terminal aberto)

```bash
conda activate pyrosetta
cd /caminho/para/dms_kleb_paper
python scripts/run_dms.py all --no-plot
```

### Servidor (em segundo plano, sessão pode ser fechada)

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pyrosetta
cd /caminho/para/dms_kleb_paper
nohup python scripts/run_dms.py all --no-plot > run.log 2>&1 &
echo $!
```

Substitua `/caminho/para/dms_kleb_paper` pelo caminho real onde o repositório foi clonado (use `pwd` para verificar).

O comando roda em segundo plano e grava tudo em `run.log`. O número impresso é o PID do processo.

## 6. Acompanhar o progresso

Ver o log em tempo real:

```bash
tail -f run.log
```

Ver quantas posições de cada proteína já foram processadas:

```bash
watch -n 30 "for p in mgrb phop pmra pmrb phoq; do \
  n=\$(ls \$p/dms_output/csv/ 2>/dev/null | wc -l | tr -d ' '); \
  echo \"\$p: \$n\"; done"
```

## 7. Tempo estimado

2 a 6 dias em um servidor moderno. O processo pode ser interrompido e retomado: ao rodar o mesmo comando novamente, ele continua de onde parou.

## 8. Resultados

Cada proteína gera um arquivo em `<nome>/dms_output/DMS_report.csv`.

Quando todas as proteínas terminarem, gerar o heatmap combinado:

```bash
python scripts/plot_combined.py
```

Os arquivos `combined_dms_heatmap.png` e `combined_dms_heatmap.pdf` serão criados na pasta principal.
