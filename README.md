概要
本プロジェクトでは、「Satou　siori」というAIモデルの開発に関わっています。このモデルは、声の変換とスタイルの変化に適応するために設計されており、高度な深層学習フレームワークを使用しています。

設定
設定ファイルconfig.jsonには、学習率、バッチサイズ、エポック数、メルチャネルやサンプリングレートなどの声の属性に関する具体的な設定が含まれています。詳細な設定は以下のconfig.jsonセクションを参照してください。

ファイルとディレクトリ
main.py: モデルの評価を実行するための音声対話メインのPythonスクリプトです。
config.json: トレーニング、データ処理、モデルのアーキテクチャに関するすべての設定を含んでいます。
style_vectors.npy: モデルで使用される、異なる声のスタイルや感情に対応したスタイルベクトルを含むNumpyファイルです。
ai.txt と ai-first.txt: AIの対話能力を様々なシナリオでテストするためのサンプル台詞やスクリプトを含むテキストファイルです。
モデルの能力
声のスタイル: 複数の声のスタイルを扱い、特定の話者の属性に条件付けが可能です。
感情認識: 様々な感情の調子で認識し、応答する能力があります。
高忠実度: 高いサンプリングレートと詳細なメルスペクトログラム設定を使用して、クリアで正確な声の出力を実現します。
使用方法
モデルのトレーニングまたは評価を開始するには：

bash
コードをコピーする
python main.py
スクリプトを実行する前に、config.jsonの設定が特定のニーズに合わせて調整されていることを確認してください。

追加情報
トークモデルバージョン: satou_e36_s1000.safetensors
開発者: [遠藤　円]
このREADMEはプロジェクトの基本的な紹介を提供します。
包括的な設定については、プロジェクトのルートディレクトリにあるconfig.jsonファイルを参照してください。






