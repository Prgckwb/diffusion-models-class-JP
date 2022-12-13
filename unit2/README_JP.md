# ユニット2：微調整、ガイダンス、条件付け

Hugging Face拡散モデルコースのユニット2へようこそ! このユニットでは、事前に学習した拡散モデルを新しい方法で使用、適応する方法を学びます。また、生成プロセスを制御するための**条件**として追加入力を受ける拡散モデルをどのように作成するのかを見ていきます。

## Start this Unit :rocket:

以下、このユニットの手順を説明します。

- 新しい教材がリリースされたときに通知されるように、[このコースにサインアップ](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162)していることを確認する。
- このユニットで重要な考え方の概要について、以下の資料に目を通してください。
- _**微調整とガイダンス**_ ノートブックをチェックして、🤗 Diffusersライブラリを使用して新しいデータセットで既存の拡散モデルを微調整し、ガイダンスを使ってサンプリング手順を変更する。
- ノートブックの例に従って、カスタムモデルのGradioデモを共有します。
- (オプション) _**条件付き拡散モデル例**_ notebookをチェックアウトして、生成プロセスにどのように追加の制御を加えることができるかを確認します。

:loudspeaker: [Discord](https://huggingface.co/join/discord)に参加するのを忘れないでください。ここでは、`#diffusion-models-class`チャンネルで教材について議論したり、作ったものを共有したりすることができます。
 
## 微調整

ユニット1で見たように、拡散モデルをゼロから学習するのは時間がかかるものです。特に高解像度になればなるほど、ゼロからモデルをトレーニングするために必要な時間とデータは現実的ではなくなります。幸いにも、解決策があります：すでにトレーニングされたモデルから始めるのです。この方法では、ある種の画像のノイズ除去をすでに学習したモデルから始めます。これは、ランダムに初期化されたモデルから始めるよりも良い出発点になることを期待しています。

![LSUN Bedroomで学習し，WikiArtで500ステップのファインチューニングを行ったモデルで生成した画像例](https://api.wandb.ai/files/johnowhitaker/dm_finetune/2upaa341/media/images/Sample%20generations_501_d980e7fe082aec0dfc49.png)

微調整は通常、新しいデータがベースモデルの元の学習データにある程度似ている場合に最も効果的ですが（例えば、漫画の顔を生成しようとする場合、顔について学習したモデルから始めるのはおそらく良い考えです）、驚くべきことに、ドメインがかなり大幅に変更されてもその効果は持続するのです。上の画像は、[LSUN Bedroomsデータセットで学習したモデル](https://huggingface.co/google/ddpm-bedroom-256)を、[WikiArtデータセット](https://huggingface.co/datasets/huggan/wikiart)で500ステップのファインチューニングを行ったものです。[学習スクリプト](https://github.com/huggingface/diffusion-models-class/blob/main/unit2/finetune_model.py)は、このユニットのノートブックと一緒に参考として添付されています。

## ガイダンス

無条件モデルは生成されるものをあまりコントロールできない。条件付きモデル（詳しくは次のセクションで説明します）をトレーニングして、追加の入力を受け取り、生成プロセスをコントロールすることはできますが、もしすでにトレーニングされた無条件モデルがあったらどうでしょうか？ガイダンスとは、生成プロセスの各ステップにおけるモデルの予測を、何らかのガイダンス関数に照らして評価し、最終的に生成される画像がより私たちの好みに合うように修正するプロセスである。

![ガイダンス画像例](guidance_eg.png)

このガイダンス機能はほとんど何でも可能であり、強力なテクニックとなる! このノートでは、単純な例（上の出力例のように色を制御する）から、CLIPという強力な事前学習済みモデルを利用し、テキスト記述に基づいて生成を誘導する例まで構築しています。

## 条件付け

ガイダンスは無条件拡散モデルからいくつかのマイルを得るための素晴らしい方法ですが、もし学習中に利用可能な追加情報（クラスラベルや画像のキャプションなど）があれば、それをモデルに与え、予測を行う際に利用することも可能です。そうすることで、**条件付き**モデルを作成し、推論時に条件付けとして入力されるものを制御することができます。ノートブックには、クラスラベルに従って画像を生成することを学習するクラス条件付きモデルの例が示されています。

![条件付け例](conditional_digit_generation.png)

このコンディショニング情報を渡すには、次のような方法がある。
- UNetへの入力に追加チャンネルとして入力する。これは条件付け情報が画像と同じ形をしている場合によく使われる。例えば、セグメンテーションマスク、深度マップ、あるいは画像のぼやけたバージョン（復元／超解像モデルの場合）などである。他のタイプの条件付けにも有効です。例えば、ノートブックでは、クラスラベルをエンベッディングにマッピングし、それを入力画像と同じ幅と高さに拡張して、追加チャンネルとして入力できるようにしています。
- エンベッディングを作り、それをunetの1つ以上の内部レイヤーの出力でチャンネル数に見合うサイズに投影し、それらの出力に追加していく。例えば、タイムステップのコンディショニングはこのように処理されます。各レスネットブロックの出力には、投影されたタイムステップのエンベッディングが追加されます。これは、CLIP画像エンベッディングのようなベクトルをコンディショニング情報として持っている場合に有効です。注目すべき例は、まさにこれを行う['Image Variations' version of Stable Diffusion](https://huggingface.co/spaces/lambdalabs/stable-diffusion-image-variations)です。
- 条件付けとして渡されたシーケンスに「参加」することができるクロスアテンション層を追加する。テキストは、変換モデルを用いてエンベッディングのシーケンスにマッピングされ、unetのクロスアテンション層は、この情報をノイズ除去パスに取り込むために使用されます。ユニット3では、安定拡散がテキストの条件付けをどのように扱うかを検証するため、これを実際に見ていきます。


## ハンズオンノートブック

| 章                                                       | Colab                                                                                                                                                                                                                   | Kaggle                                                                                                                                                                                                   | Gradient                                                                                                                                                                               | Studio Lab                                                                                                                                                                                                   |
|:--------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 微調整とガイダンス                                               | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              | [![SageMaker Studio Labで開く](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit2/01_finetuning_and_guidance.ipynb)              |
| クラス条件付き拡散モデル例 | [![Colabで開く](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              | [![SageMaker Studio Labで開く](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb)              |

この時点で、付属のノートブックを使い始めるのに十分な知識があります。上記のリンクから好きなプラットフォームで開いてみてください。ファインチューニングはかなり計算量が多いので、KaggleやGoogle Colabを使っている場合は、ランタイムタイプを「GPU」に設定しておくとよいでしょう。

この資料の大部分は、_**微調整とガイダンス**_ にあり、実例を通してこの2つのトピックを探求しています。このノートブックでは、新しいデータで既存のモデルをファインチューニングし、ガイダンスを追加し、その結果をGradioのデモとして共有する方法を示しています。付属のスクリプト([finetune_model.py](https://github.com/huggingface/diffusion-models-class/blob/main/unit2/finetune_model.py))では、様々なファインチューニングの設定を簡単に試すことができ、🤗 Spacesであなた自身のデモを共有するためのテンプレートとして使える[an example space](https://huggingface.co/spaces/johnowhitaker/color-guided-wikiart-diffusion) が用意されています。

_**クラス条件付き拡散モデルの例**_ では、MNISTデータセットを用いてクラスラベルを条件とした拡散モデルを作成する簡単な作業例を紹介します。モデルにノイズ除去のための情報を与えることで、推論時にどのような画像が生成されるかを制御することができます。

## プロジェクトタイム

_**微調整とガイダンス**_ ノートブックの例に従って、自分のモデルを微調整するか、既存のモデルを選び、Gradioデモを作成して、あなたの新しいガイダンススキルを披露してください。デモを Discord や Twitter などで共有し、あなたの作品を賞賛することを忘れないでください。

## その他の資料

[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) - DDIMサンプリングメソッドを導入（DDIMSchedulerで使用）。
 
[GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741) - 拡散モデルをテキストに条件付けする手法を導入

[eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers](https://arxiv.org/abs/2211.01324) - さまざまな条件付けを併用することで、生成されるサンプルをより自在にコントロールすることが可能です。

もっと素晴らしいリソースがありますか？このリストに追加します。
