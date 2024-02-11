# MVC
Projeto para resolução do problema para o edital FIESC/00122/2024 - Pesquisador I - Inteligência Artificial

* 0_study.ipynb
	* Análise exploratória dos dados

* 1_dataset_prep.ipynb
	* Processamento de todos os dados, gerando as features e separando os conjuntos de treino, validação e teste

* 2_Dmodel.ipynb
	* Processo de definição do modelo e treinamento

* 3_test_Dmodel.ipynb
	* Avaliação do modelo treinado

* run_case.py
	* Script com um exemplo de processo de inferência utilizando os sinais dos sensores
	* $python run_case.py --model_f ./model.checkpoint --sample ./sample.json --feature_norm ./norm_feat_params.json --freq_norm ./norm_freq_params.json

* sample.json
	* Um arquivo contendo uma única sample para teste

* mls/
	* diretório com definições de modelo e funções para treino e validação

* utils/
	* algumas implementações para o processamento dos sinais, plots, e processamento de dados utilizados na inferência
	
* reports/
	* arquivo contendo a execução dos jupyters em formato .html

* IMPORTANTE: Os scripts .ipynb possuem uma flag no início de cada página
    * True faz com que os dados sejam salvos ou o treinamento seja executado
    * False ira executar todas as células, mas sem salvar dados ou carregando arquivos quando for o caso 