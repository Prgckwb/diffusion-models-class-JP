# ユニット1: 拡散モデル入門

Hugging Face 拡散モデル講座のユニット1へようこそ！
このユニットでは、拡散モデルの仕組みの基本を学び、🤗Diffusersライブラリを使用して独自のモデルを作成する方法について学びます。

## このユニットを開始する :rocket:

以下、このユニットの手順を説明します。

- 新しい教材がリリースされたときに通知されるように、このコースに[サインアップ](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162)
していることを確認してください
- 以下の紹介資料と、興味のありそうな追加資料に目を通す
- 以下の _**Diffusers入門**_ ノートブックで、🤗Diffusersライブラリを使った理論の実践をチェックする
- ノートブックまたはリンクされたトレーニングスクリプトを使用して、独自の拡散モデルを学習し、共有する
- (オプション) _**ゼロから始める拡散モデル**_ ノートブックで、最小限の実装を確認し、設計上のさまざまな思考に興味がある場合は、さらに深く掘り下げる

:loudspeaker: [Discord](https://huggingface.co/join/discord)
に参加するのを忘れないでください。この教材について議論したり、作ったものを `#diffusion-models-class` チャンネルで共有したりすることができます。

## 拡散モデルとは？

拡散モデルは、「生成モデル」として知られるアルゴリズム群に比較的最近追加されたものです。
生成モデリングの目標は、多くの学習例が与えられたときに、画像や音声などのデータを**生成**するように学習することです。
優れた生成モデルは、学習データをそのままコピーすることなく、
それに類似した**多様な**出力セットを作成することができます。
拡散モデルはどのようにしてこれを実現するのでしょうか？
ここでは、説明のために画像生成のケースに焦点を当てましょう。

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png" width="800" alt=""/>
    <br>
    <em> DDPM論文からの図 (https://arxiv.org/abs/2006.11239). </em>
<p>

拡散モデルの成功の秘密は、拡散プロセスの反復性にあります。
生成はランダムなノイズから始まりますが、出力画像が現れるまで、
何段階にもわたって徐々に洗練されていきます。各ステップにおいて、
モデルは現在の入力から完全にノイズ除去されたバージョンまでどのように進むかを推定します。
しかし、各ステップで小さな変更を加えるだけなので、
初期段階（最終的な出力を予測することが非常に難しい）でのこの推定値の誤差は、
後の更新で修正することができます。

モデルの学習は、他の種類の生成モデルに比べて比較的簡単である。私たちは繰り返し

1) 学習データから画像を読み込む
2) 様々な量のノイズを加える。ここで私たちは、『限りなくノイズに近い画像と完璧に近い画像の両方をどのように「修正」（ノイズ除去）するかについて、モデルに良い仕事をしてほしい』という事を忘れないでください
3) 入力にノイズを含んだものをモデルに与える
4) これらの入力に対して、モデルがどの程度ノイズ除去ができるかを評価する
5) この情報をもとに、モデルの重みを更新する

学習されたモデルを使って新しい画像を生成するには、まず完全にランダムな入力から始めて、
それを繰り返しモデルに与え、モデルの予測に基づいて毎回少しずつ更新していきます。
後述するように、このプロセスを効率化し、できるだけ少ないステップで良い画像を生成できるようにするためのサンプリング手法が数多く存在する。

ユニット1では、これらの各ステップをハンズオンノートブックで詳しく紹介します。ユニット2では、このプロセスをどのように変更し、追加の条件付け（クラスラベルなど）やガイダンスなどの手法によって、モデルの出力にさらなる制御を加えることができるかを見ていきます。そしてユニット3と4では、テキストの説明文から画像を生成することができる、Stable
Diffusionと呼ばれる非常に強力な拡散モデルについて検討する。

## ハンズオンノートブック

この時点で、付属のノートブックに取り掛かるのに十分な知識があることになります！
この2つのノートは、同じアイデアを異なる方法で表現しています。

| 章                   | Colab                                                                                                                                                                                                       | Kaggle                                                                                                                                                                                                                | Gradient                                                                                                                                                                                            | Studio Lab                                                                                                                                                                                                           |
|:--------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Diffusers入門         | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)     | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)     | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)     | [![SageMaker Studio Labで開く](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb)     |
| ゼロから始めるDiffusionモデル | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb) | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb) | [![SageMaker Studio Labで開く](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb) |

_**Diffusers入門**_ では、diffusersライブラリのビルディングブロックを使用して、上記の様々なステップを紹介します。
どのようなデータであっても、独自の拡散モデルを作成し、学習し、サンプリングする方法をすぐに理解することができます。
このノートブックの終わりには、サンプルの学習スクリプトを読んで修正し、拡散モデルを学習し、世界と共有することができるようになります。
このノートブックはまた、このユニットに関連した主な演習を紹介します。ここでは、様々なスケールの拡散モデルのための良い「学習レシピ」を共同で見つけようとします。
詳細は次のセクションを参照してください。

_**ゼロから始めるDiffusers入門**_では、同じステップ（データへのノイズの追加、モデルの作成、トレーニング、サンプリング）をPyTorchで一から実装し、できる限りシンプルに表示しています。
そして、この「おもちゃの例」をdiffusersのバージョンと比較し、
両者がどのように違うのか、どこが改善されたのかを指摘します。
ここでのゴールは、異なるコンポーネントとそこに込められた設計上の決定に慣れ、新しい実装を見るときに、重要なアイデアを素早く識別できるようにすることです。

## プロジェクトタイム

さて、基本を押さえたところで、1つまたは複数の拡散モデルを学習してみてください。
いくつかの提案は、_**Diffusers入門**_ ノートブックの最後に記載されています。
あなたの結果、学習レシピ、発見をコミュニティと共有し、
これらのモデルを学習するための最良の方法を一緒に考えましょう。

## その他の資料

[注釈付き普及モデル](https://huggingface.co/blog/annotated-diffusion)
は、DDPMの背後にあるコードと理論について、数学とコードですべての異なる構成要素を示しながら、非常に詳細なウォークスルーとなっています。また、さらに読み進めるために多くの論文にリンクしています。

[条件なし画像生成](https://huggingface.co/docs/diffusers/training/unconditional_training)
のHugging Faceのドキュメント に、公式の学習例スクリプトを用いた拡散モデルの学習方法の例と、
独自のデータセットを作成する方法を示すコードが掲載されています。

Diffusionモデルについての AI コーヒーブレイク動画: https://www.youtube.com/watch?v=344w5h24-h8

Yannic KilcherのDDPMの動画: https://www.youtube.com/watch?v=W-O7AZNzbzQ

もっと素晴らしいリソースがありますか？このリストに追加します。
