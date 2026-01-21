# Wrap-up Report : 영화 감성 분석

작성자 : 구승민_T8015

### 1. 프로젝트 개요

**▶ 1-1. 프로젝트 주제**

- **한국어 영화 리뷰** 바탕, 텍스트가 **긍정적인지 부정적인지** 분류하는 감성 분류 모델
- **Label = { 0 : '강한 부정', 1 : '약한 부정', 2 : '약한 긍정', 3 : '강한 긍정' }**
- 140자 미만의 짧은 문장으로 구성
- 학습 데이터는 **LLM 활용 합성 데이터 일부 포함**
- 사람이 판단하기 어려운 문장이나 애매한 문맥을 포함하는 고난이도 샘플 포함

**▶ 1-2. 프로젝트 구현 내용, 컨셉, 교육 내용과의 관련성 등**

- 프로젝트 최종 결과물은 총 2개의 모델 선정
    - TAPT 된 3개의 **`klue/roberta-base`, `klue/bert-base`,**  **`monologg/koelectra-base-v3-discriminator`** 모델을 파인 튜닝 후 soft voting한 앙상블 모델
        - Accuracy : Public 0.8293, Private 0.8293
    - **`kykim/bert-kor-base`** 모델에 K-Fold(K=5) 교차 검증으로 파인 튜닝을 한 모델
        - Accuracy : Public 0.8253, Private 0.8255

**▶ 1-3. 활용 장비 및 재료 (개발 환경, 협업 tool 등)**

- OpenSSL로 VPN이 구성된 **OpenVPN** 활용, 접근 시 **VS Code**의 **Remote-SSH** 활용한 개발 환경 구축
- **V100** GPU 활용 : **RAM 32GB**, F16 기준 120 TFLOPS 이상
- 모델 : **`Transformers`**, 파인 튜닝 : **`Trainer API`**, 파라미터 최적화 : **`Optuna`**, 모니터링 : **`wandb`**

### 2. 프로젝트 수행 절차

**▶ 2-1. 프로젝트 기획**

다음과 같은 수행 프로세스를 기획함

1. **환경 구축** 및 가상 공간 활용
2. **베이스라인 코드 실행** 및 이해
3. 추가적 **탐색 아이디어 구현 및 시각화**
4. 베이스라인 코드 기준 **최적 LLM 모델 탐색**
5. 도메인에 맞는 **전처리 프로세스 구축** 또는 변경
6. 학습 과정 변경 및 비교 : **KFold**, **Ensemble** 등
7. 최적 하이퍼파라미터 탐색 : **Optuna**
8. 결과 비교 후 **최적 모델 탐색** 및 재학습

**▶ 2-2. 프로젝트 수행 과정**

<aside>

1. 가상 환경 설정

<img width="574" height="117" alt="1" src="https://github.com/user-attachments/assets/4550976e-b274-42be-8770-a792d64c8ced" />


SSH로 접속한 환경에서 `root` 디렉토리는 용량이 부족한 상태

- `/data/ephemeral` 하위에서 수행할 수 있도록 경로 변경
    - 아래 명령어를 통해 해당 쉘에 스크립트 즉시 적용
    - `source /data/ephemeral/home/py310/bin/activate`
</aside>

<aside>

1. 베이스라인 코드
- ML 워크플로우
    - 데이터 준비 → EDA → 전처리 → 데이터 분할 → 모델 설정 → 훈련 및 평가
- 코드 실행 결과 Accuracy 0.8074
    
  <img width="412" height="72" alt="2" src="https://github.com/user-attachments/assets/e6148f30-ac45-4ccc-bb6e-43bd67f9bbaa" />

    
</aside>

<aside>

1. 탐색 아이디어 구현 및 시각화
- 전처리 변경
    - clean_text 메서드의 “ㅋㅋ”과 “ㅠㅠ”가 전처리 되지 않도록 수정
    - Regex의 인자 값의 범위에서 “ㅋㅋ”과 “ㅠㅠ” 부분을 제외
    
    ```python
    def _clean_text(self, text):
    		if pd.isna(text):
    		    return ""
    		
    		text = str(text).strip()
    		text = re.sub(r"[ㄱ-ㅊㅌㅍㅏ-ㅛㅡ-ㅣ]+", "", text)
    		text = re.sub(r"([ㅋㅎ])\1{2,}", r"\1\1", text)
    		text = re.sub(r"([ㅠㅜㅡ])\1{2,}", r"\1\1", text)
    		text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    		text = re.sub(r"[^\w\s가-힣.,!?ㅋㅎㅠㅜㅡ~\-]", " ", text)
    		text = re.sub(r"\s+", " ", text)
    		
    		return text.strip()
    ```
    
- 데이터 시각화
    - wordcloud를 이용하여 어떤 단어가 많이 등장하는지 시각화
    
    ```python
    from wordcloud import WordCloud
    visualize_data = [review for review in df["review"] if type(review) is str]
    wordcloud = WordCloud("../font/NanumGothic.ttf").generate(''.join(visualize_data))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
    ```
    
  <img width="1550" height="332" alt="3" src="https://github.com/user-attachments/assets/1d6c2ed9-d081-4df9-aa7f-aaf2e6e33542" />

    
</aside>

<aside>

1. 최적 LLM 모델 탐색
- 허용된 5개의 모델 중 기본 성능이 높은 모델 탐색
    - 하이퍼파라미터 변동 없이 모델만 변경 후 결과 비교
    - 가장 높은 점수의 `kykim/bert-kor-base` 모델 활용 : Accuracy 0.8208
        
      <img width="432" height="79" alt="4" src="https://github.com/user-attachments/assets/c8f26ced-a6d4-419e-a151-6455c21da3b2" />

        
</aside>

<aside>

1. 전처리 프로세스 구축
- 시각화 이후 가장 많이 보이는 상위 3개의 단어를 `Special Token` 처리

```python
MOVIE1_RE = re.compile(r"이 영화는")
MOVIE2_RE = re.compile(r"이 영화")
MOVIE3_RE = re.compile(r"보는 내내")
SPECIAL_MAP = {
    "MOVIE_REF": "[MOVIE_REF]",
    "DURING_WATCH": "[DURING_WATCH]",
}
def simple_preprocess(text: str) -> str:
    text = MOVIE1_RE.sub(SPECIAL_MAP["MOVIE_REF"], text)
    text = MOVIE2_RE.sub(SPECIAL_MAP["MOVIE_REF"], text)
    text = MOVIE3_RE.sub(SPECIAL_MAP["DURING_WATCH"], text)
    return text
special_tokens_dict = {"additional_special_tokens": list(SPECIAL_MAP.values())}
tokenizer.add_special_tokens(special_tokens_dict)
```

- 토큰 처리 이후 성능 상향 결과
    - 결과 비교 시 **0.8208** → **0.8212 (+0.0004)** 소폭 증가 확인
        
      <img width="461" height="89" alt="5" src="https://github.com/user-attachments/assets/f48a0439-cc2d-4f52-81b4-d199c64bd372" />

        
</aside>

<aside>

1. 학습 과정 변경 및 비교
- Epoch 5 → 10
    - Accuracy **0.8208** → **0.8144 (-0.0064)** 소폭 감소 확인
        
      <img width="471" height="77" alt="6" src="https://github.com/user-attachments/assets/8ee263f2-a826-43af-8e98-6c1aa68ea7c4" />

        
- K-Fold (K=5) 교차 검증 적용
    
    ```python
    # K-Fold 교차 검증
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []
    
    # 훈련 실행
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # 전처리 및 데이터셋 생성
        X_train_processed = preprocessor.fit_transform(X_train_fold.tolist(), y_train_fold.tolist())
        X_val_processed = preprocessor.transform(X_val_fold.tolist())
        
        train_dataset = ReviewDataset(pd.Series(X_train_processed), y_train_fold, tokenizer, CHOSEN_MAX_LENGTH)
        val_dataset = ReviewDataset(pd.Series(X_val_processed), y_val_fold, tokenizer, CHOSEN_MAX_LENGTH)
    
        # Trainer 초기화
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
    
        try:
            training_results = trainer.train()
    
            # 훈련 로그 정보 출력
            if hasattr(training_results, "log_history"):
                print(f"총 훈련 스텝: {training_results.global_step}")
                
            # print("최종 검증 성능 평가 중...")
            eval_results = trainer.evaluate()
    
        except KeyboardInterrupt:
            print("\n사용자에 의해 훈련이 중단되었습니다.")
            raise
        except Exception as e:
            print(f"\n오류 발생: {str(e)}")
            raise
    ```
    
    - Accuracy **0.8208** → **0.8253 (+0.0045)** 소폭 증가 확인
        
      <img width="406" height="79" alt="7" src="https://github.com/user-attachments/assets/ed2f27ea-9c79-4bbf-9cbb-0517d465ce78" />

        
- Ensemble Soft-Voting 적용
    
    ```python
    # 각 모델 훈련 루프
    for model_name in model_names:
        trainer, tokenizer, model = train_model(model_name)
        trainers.append(trainer)
        tokenizers.append(tokenizer)
        models.append(model)
    
    # 테스트 데이터 처리 및 앙상블 추론
    df_test = pd.read_csv("../data/test.csv")
    test_texts = df_test["review"].tolist()
    test_processed = preprocessor.transform(test_texts)
    test_processed = pd.Series(test_processed)
    
    test_datasets = []
    for i, tokenizer in enumerate(tokenizers):
        test_datasets.append(ReviewDataset(test_processed, None, tokenizer, CHOSEN_MAX_LENGTH))
    
    all_logits = []
    for i, trainer in enumerate(trainers):
        pred = trainer.predict(test_datasets[i])
        all_logits.append(pred.predictions)
    
    avg_logits = np.mean(all_logits, axis=0)
    predicted_labels = np.argmax(avg_logits, axis=1)
    
    df_test["pred"] = predicted_labels
    ```
    
    - Accuracy **0.8253** → **0.8293 (+0.0004)** 소폭 증가 확인
        
      <img width="423" height="81" alt="8" src="https://github.com/user-attachments/assets/772461ce-0a5d-40c3-bfb9-fb6f6fb0cfaa" />

        
</aside>

<aside>

1. 최적 하이퍼파라미터 탐색
- Optuna 최적 하이퍼 파라미터 찾는 `objective` 함수 정의
    
    ```python
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
        num_epochs = trial.suggest_int("num_epochs", 2, 5)
        batch_size_train = trial.suggest_categorical("batch_size_train", [64, 128, 256])
        batch_size_eval = trial.suggest_categorical("batch_size_eval", [64, 128, 256])
        warmup_steps = trial.suggest_int("warmup_steps", 0, 1000)
        weight_decay = trial.suggest_float("weight_decay", 0.001, 0.1, log=True)
        max_length = trial.suggest_categorical("max_length", [64, 128])
    
        # 전체 데이터 전처리
        X_processed = preprocessor.fit_transform(X.tolist(), y.tolist())
        X_processed = pd.Series(X_processed)
    
        X_train, X_val, y_train, y_val = train_test_split(X_processed, y,test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
        train_dataset = ReviewDataset(X_train, y_train, tokenizer, max_length)
        val_dataset = ReviewDataset(X_val, y_val, tokenizer, max_length)
    
        model = AutoModelForSequenceClassification.from_pretrained("../kor-bert", num_labels=NUM_CLASSES)
    
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size_train,
            per_device_eval_batch_size=batch_size_eval,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            seed=RANDOM_STATE,
            fp16=torch.cuda.is_available(),
        )
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
    
        trainer.train()
        eval_results = trainer.evaluate()
    
        return eval_results["eval_f1"]
    
    # 캐시 초기화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Optuna 실행
    tokenizer = AutoTokenizer.from_pretrained("../kor-bert")  # Optuna용 토크나이저
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_roberta, n_trials=10)
    best_params_roberta = study.best_params
    print(f"최적 하이퍼파라미터: {best_params_roberta}")
    ```
    
    - Accuracy **0.8293** → **0.8235 (-0.0058)** 최고점 대비 소폭 감소 확인
        
      <img width="423" height="81" alt="9" src="https://github.com/user-attachments/assets/c37f4556-7440-4dc2-9413-b386fea1a597" />

        
</aside>

**▶ 2-3. 프로젝트 수행 결과**

|  | Base-Line | BERT-kor-base | Special Token | K-Fold 5 | Soft Voting | Optuna |
| --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 0.8074 | 0.8208 | 0.8212 | **0.8253** | **↑ 0.8293** | 0.8235 |

**▶ 2-4. 기간 및 활동 내용 정리**

- 실제 개발 기간 : `25.10.27 - 25.10.30` (총 4일)
    - `10.27` : 환경 구축 및 코드 실행
    - `10.28` : Special Token 적용, 최적 모델 탐색, K-Fold 적용
    - `10.29` : Optuna 및 Ensemble 적용
    - `10.30` : 결과 비교 및 최적 모델 탐색

### 3. 자체 평가 의견 및 회고

**▶ 3-1. 자체 평가 의견**

<aside>

초기 목표는 베이스라인 코드보다 높은 성능을 내는 것이었습니다. 모델 변경 만으로도 결과가 좋게 나와서 여러 기법을 적용하는 방식으로 접근 방식을 변경하였습니다. K-Fold 교차 검증에서 예상보다 결과가 좋았기 때문에 개인적으로 계획보다는 결과가 잘 나왔습니다.

</aside>

**▶ 3-2. 잘한 부분과 아쉬운 점을 작성**

- 느낀점
    
    <aside>
    
    여러 기법을 활용할 수 있어서 좋았습니다. 아직은 부족하고 열심히 깨지면서 성장한다고 생각합니다. 이제 시작이라고 생각하고 더 나아갈 방법은 많이 있기 때문에 즐거운 마음으로 임했던 것 같습니다. 예상보다 결과가 좋게 나왔기 때문에 그 의도를 다시 파악하는데 집중하고 여러 기법을 반복 적용하면서 하드 스킬을 늘리는데 중점을 둘 것 같습니다.
    
    </aside>
    
- 잘한 점들
    
    <aside>
    
    빠른 구현이 좋았습니다. 완벽히 이해하지 못 하더라도 일단 만들면서 경험을 하면 그 구조가 무의식적으로 익혀진다고 생각합니다. 그 내부에서 변수를 조정하면서 어떤 결과가 일어나는지 직접 체감하여 한 걸음 더 나아간 부분이 잘 했다고 생각합니다.
    
    </aside>
    
- 시도 했으나 잘 되지 않았던 것들
    
    <aside>
    
    - Optuna를 활용한 하이퍼파라미터 최적화
        - 최적값을 찾는 시간이 오래 걸린 탓에 n_trial을 최대 10으로 밖에 주지 못했습니다.
        - 보통 20~50의 값으로 최적값을 찾는데, 해당 시간만큼 충분한 시간을 할애하지 못했기 때문이라고 생각합니다.
    - 전처리 과정 개선
        - 스페셜 토큰이 효과가 보였기 때문에 상위 10개로 늘렸지만, 이는 오히려 성능이 감소하는 효과를 가져왔습니다.
    </aside>
    
- 아쉬웠던 점들
    
    <aside>
    
    - 모델 저장 미활용
        - 인자 중 model save 기능이 있었는데, 중요성을 알지 못하고 활용하지 못했습니다.
        - 비슷한 모델의 다른 결과를 가져오기 위해 다시 학습 시키는 과정은 시간 낭비였고, 이를 모델 저장으로 아끼지 못했던 부분이 아쉬웠습니다.
    - 중복 제거 및 증강 데이터 제거
        - 이 과정을 통해 더 정규화를 할 수 있다는 생각이 들었지만, 앙상블 및 K-Fold 교차 검증에 많은 시간이 들어 이를 활용하지 못한 점이 아쉬웠습니다.
    </aside>
    
- 프로젝트를 통해 배운 점 또는 시사점
    
    <aside>
    
    모델 빌딩은 어려운 과정이지만 최적값을 찾아가는 과정은 즐겁게 느껴졌습니다. 베이스라인 코드를 통해 해당 과정을 빠르게 넘어감으로써 생각해본 아이디어들을 구현하는 것이 좋았고, 후반부에 어떤 적용법을 더 활용할지 몰라 다른 캠퍼에게 어떤 방식으로 해결하고 있는지 서로 의견 공유를 하며 개선해 나간 점이 좋았습니다. 현재 팀의 피어 세션과 이전 팀과의 슬랙 채팅을 통해 네트워킹을 하며 서로 발전해나감을 느끼는 것이 매우 인상 깊었습니다.
    
    </aside>
