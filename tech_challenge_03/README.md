# FIAP - Curso IA para Devs
# Tech Challenge 03 
# Problema: 
  -- No Tech Challenge desta fase, você precisa executar o fine-tuning de um 
  -- foundation model (Llama, BERT, MISTRAL etc.), utilizando o dataset "The
  -- AmazonTitles-1.3MM".  O modelo treinado deverá:

● Receber perguntas com um contexto obtido por meio do arquivo json
“trn.json” que está contido dentro do dataset.

● A partir do prompt formado pela pergunta do usuário sobre o título do
produto, o modelo deverá gerar uma resposta baseada na pergunta do
usuário trazendo como resultado do aprendizado do fine-tuning os
dados da sua descrição.

# Grupo 44
- Francisco Antonio Guilherme
- fagn2013@gmail.com

- Marcelo Lima Gomes
- marcelolimagomes@gmail.com

- FELIPE MORAES DOS SANTOS
- felipe.moraes.santos2014@gmail.com

## Descrição do processo de seleção e preparação do dataset:

** >> Vídeo do projeto: https://drive.google.com/file/d/1qYxyD9Azbamqx3ly0xTUgAJaO3m3S2Ul/view?usp=sharing

- Foi escolhido o dataset "The AmazonTitles-1.3MM" conforme orientado no enunciado do Tech Challenge 03.
- O processo de preparação foi aplicado.
 * Conversão dos caracteres especiais no padrão HTML para Texto Unicode utilizando a biblioteca python "html";
 * Remoção dos registros utilizando os seguintes critérios:
   1. Titulo com tamanho == 0;
   2. Descrição menores que 400 caracteres e maiores que 500 caracteres.
   3. Remoção de espaços em branco desnecessários. 
 * Obs.: Fizemos a simulação de remoção de pontuação e outros caracteres que, inicialmente julgamos que poderia atrapalhar o modelo, porém percebemos que o modelo convergia melhor com eles, então foram mantidos. Foram testados a remoção dos seguintes caracteres: -\",;:_()
 * O processo de tratamento dos dados foi implementado no arquivo "limpeza_dataset.ipynb";
 
 - Para o processo de treinamento foi escolhido o modelo "unsloth/tinyllama" com quantização de 16bits;
 - Utilizamos uma GPU Nvidia RTX 3060 12GB e o ambiente foi configurado utilizando o  anaconda. Foi utilizado a versão 3.10 do python as bibliotecas cuda versão 12.1;
 - Utilizamos a API do Unsloth para executar o processo de treinamento, porém aplicamos o treinamento com a quantização em 16 bits, ajustado o parâmetro 'load_in_4bit=False';
 - Foi utilizado o prompt no padrão Alpaca Prompt incluindo a ### Instrução "Write a book review.";
 - Utilizamos um 'batch size de treinamento = 128' para otimizar a alocação de memória da GPU, pois valores menores observamos que o processo não consumia toda memória da GPU, assim ocasionando em maior "I/O" de memória e menos performance. O parâmetro ajustado foi 'per_device_train_batch_size=128';
 - Optamos por reduzir o dataset de treino, pois em simulações utilizando o dataset integral, o processo para aplicar 1 época de treino iria demorar 33 horas, então reduzimos o dataset para 10.000 registros.
 - Foram executados dois ciclos de treinamento:
    * 1o Ciclo foram utilizados 10 épocas de treinamento o que consumiu aprox. 3,5 horas de treino e 730 steps;
    * 2o Ciclo foram utilizados 30 épocas de treinamento o que consumiu aprox. 10 horas de treino e 2370 steps.
  - Após 2 ciclos de treno, +10h de treino e 3100 steps pudemos constatar que o modelo convergiu muito bem.
  - O processo de treinamento foi implementado no arquivo "unsloth - treinar_tinyllama.ipynb" e o processo de geração de textos foi implementado no arquivo: "unsloth - teste_tinyllama.ipynb".