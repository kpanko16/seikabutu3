'''
今回は5つの機能を考えました。
1, TFIDFによる検索機能です。内容は、検索した単語が歌詞の中の重要度の高い位置付けとなっている楽曲を表示します。
また、検索ワードと関連性の高い３つの単語も共に検索します。メリットは、例えば夏と検索した時に、夏だけでなく海などの
単語も検索されより幅広く曲を検索できます。夏っぽくても夏という歌詞が入っているとは限らないのでこちらを搭載することを
考えました。今回は歌詞内の単語の検索しかできないので、今後は歌詞からも同義語を照合対象にするなどして、より柔軟性を
高くしていきたいと思います。

2, 歌詞の類似度検索機能です。各歌詞のベクトルをFAISSで演算を行い似た歌詞を持つと判定されたものを鑑賞できるよう
記載します。上下どの機能を使って移ったページからも、曲名を押すことでこの機能を使ったページに移るという仕組みになっています。

3, 音楽再生、歌詞表示機能です。仮にアプリに載っていなかった楽曲があった場合に自分でファイルを載せて再生して、音声認識で
歌詞表示する機能です（こちらは何かメリットというよりただ音声認識のコードを書いてみたかっただけです）。

4, 再生履歴のデータをもとに、ユーザー好みの曲を分類してレコメンドする機能です。基本は楽曲と歌詞の情報をもとに分類しており、
アプリを操作するときにレコメンドのボタンを一度押すとどのようなグループ分けになっているのかを数値で見ることができるように
なっています。数値は、グループ分けされた楽曲の項目ごとの値の平均値です。メリットは曲調と歌詞両方の観点での客観的なデータを
ユーザーが自ら見て、適切な楽曲グループを選択して聴くことができる点だと考えます。

5, 再生履歴データをもとに、次にどのような楽曲を聴きたいかを時系列で追っていくという考えをもとに推測してレコメンドする機能です。
こちらはまとまった時間で音楽を聴きたいと思った時に使える機能だと考えます。メリットはユーザーがわざわざ選択しなくても自動で
気分にあった曲を聴き続蹴ることが期待できる点です。今回は時間が足りず、単純に時系列で特徴量を分析するということで終わって
しまいました。今後の施策として考えられることは、複数のユーザーのデータをロードして、似た傾向のユーザの曲の選定を基準に
曲そのものをラベルとしてレコメンドする、深層強化学習やGNNによって楽曲とユーザの関係を学習して長期的に反映することなどが
考えらえます。
'''
import csv
import ctypes
import os
import pickle
import threading
import faiss
import librosa
import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import whisper
from flask import Flask
from flask import request, jsonify, render_template
from scipy.sparse import csr_matrix, hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sudachipy import dictionary
from sudachipy import tokenizer
import time
from train_and_evaluate import LSTM  # LSTMモデルをインポート
from collections import deque

app = Flask(__name__)

# Sudachiの形態素解析器を設定
tokenizer_obj = dictionary.Dictionary().create()

def sudachi_tokenizer(text):
    """
    テキストを形態素解析し、基本形をリストとして返す関数
    
    Parameters:
    text: 解析対象のテキスト
    
    Returns:
    基本形のリスト（名詞、形容詞、動詞のみ）
    """
    # テキストを形態素解析し、基本形をリストとして返す
    tokens = [m for m in tokenizer_obj.tokenize(text, tokenizer.Tokenizer.SplitMode.C)]
    # 名詞、形容詞、動詞のみを抽出し、基本形を取得
    filtered_tokens = [m.dictionary_form() for m in tokens if m.part_of_speech()[0] in ['名詞', '形容詞', '動詞']]
    return filtered_tokens

# 履歴を保存するファイル
HISTORY_FILE = 'history.csv'
MAX_HISTORY_SIZE = 10000

# 履歴を保存する関数
def save_history(row):
    # 履歴を読み込む→リスト化
    try:
        with open(HISTORY_FILE, 'r', newline='') as file:
            reader = csv.reader(file)
            history = list(reader)
    except FileNotFoundError:
        history = []

    # タイムスタンプを追加
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    row_with_timestamp = [timestamp] + row

    # 履歴を追加
    history.append(row_with_timestamp)

    # 履歴がMAX_HISTORY_SIZEを超えた場合、古いものから削除
    if len(history) > MAX_HISTORY_SIZE:
        history = history[-MAX_HISTORY_SIZE:]

    # 履歴を保存
    with open(HISTORY_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(history)

# 歌詞データCSV
df = pd.read_csv('comdata.csv') #faiss用
dfe = df.copy() #cluster用↓これらは固有のものとなるので削除
#このアーティストはこのような曲調にこのような曲調になりがちみたいな偏見を避けて純粋に局情報に集中させる目的で消去
dfe = dfe.drop(columns=['曲名', '歌い出し', 'name', 'album', 'artist', 'popularity'])

# 歌詞の単語数を計算
word_counts = []
for lyrics in df['歌い出し']:
    tokens = sudachi_tokenizer(lyrics)
    word_counts.append(len(tokens))

# 単語数の列を追加
dfe['words'] = word_counts

#javascriptで履歴のコードを書いた
@app.route('/save_history/<int:song_index>', methods=['POST'])
def save_history_endpoint(song_index):
    # 履歴を保存
    track_features = dfe.iloc[song_index].to_list()
    save_history(track_features)
    
    # 時系列モデルの微調整
    on_play(track_features)
    print("tuning finished")
    return jsonify({'status': 'success'})

#共起行列データ（クラスタリングとは独立して使用）
dfc = pd.read_csv('co_occurrence_matrix_lylics.csv', header=None, names=['Word1', 'Word2', 'Count'])
dfc['Count'] = pd.to_numeric(dfc['Count'], errors='coerce')

# TF-IDFベクトル化器を作成、形態素解析器はsudachi、基本形返す
vectorizer = TfidfVectorizer(tokenizer=sudachi_tokenizer)

# 歌詞データに対して形態素解析を行い、TF-IDF行列を作成
tfidf_matrix = vectorizer.fit_transform(df['歌い出し'].apply(lambda x: ' '.join(sudachi_tokenizer(x))))
#0でない → 各ドキュメントにおいて何種類の単語が存在するか
non_zero_counts = np.count_nonzero(tfidf_matrix.toarray(), axis=1)
dfe['words'] = pd.DataFrame(non_zero_counts, columns=['NonZeroCount'])

# vecsをNumpy→ノルムで割って正規化→faiss index作成→ベクトルをインデックスに追加
with open('vecs.pkl', 'rb') as f:
    vecs = pickle.load(f)
vecs_array = np.array(vecs).astype("float32")
vecs_array /= np.linalg.norm(vecs_array, axis=1)[:, np.newaxis]
index_f = faiss.IndexFlatIP(768)  # BERT(SimCSE)は768次元
index_f.add(vecs_array)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_word = request.form.get('search_word', None)
        if search_word:
            # 単純な検索
            title_matches = df[df['曲名'] == search_word]
            title_results = [(row['曲名'], row['artist'], row.name, 'dummy_audio.wav', '歌詞') for _, row in title_matches.iterrows()]
            # 歌詞照合検索
            filtered_df = dfc[dfc['Word1'] == search_word]
            top3_rows = filtered_df.nlargest(3, 'Count')
            values_d = top3_rows['Word2'].tolist()
            values_d.append(search_word)
            search_words = values_d

            tfidf_values_combined = []
            for s in search_words:
                word_index = vectorizer.vocabulary_.get(s)
                if word_index is not None:
                    word_tfidf_values = tfidf_matrix.getcol(word_index).toarray().flatten()
                    word_tfidf_sparse = csr_matrix(word_tfidf_values).T
                    tfidf_values_combined.append(word_tfidf_sparse)

            if tfidf_values_combined:
                tfidf_combined_matrix = hstack(tfidf_values_combined)
                top10 = (tfidf_combined_matrix.sum(axis=1).A1 / dfe['words']).argsort()[-10:][::-1]
                results = []
                for idx in top10:
                    song_name = df.iloc[idx]['曲名']
                    if song_name != search_word:
                        results.append((song_name, df.iloc[idx]['artist'], idx, 'dummy_audio.wav'))
            else:
                results = []

            final_results = title_results + results
            return render_template('resultsaudio.html', search_word=search_word, results=final_results)

    return render_template('indexhightext.html')

# ここから音声認識
"""
Flask + faster-whisper
15 秒ずつ分割 → ストリーム認識 → SSE でチャンク送信
MP3 / WAV / FLAC をサポート
"""

import os, subprocess, tempfile, threading, queue, time, ctypes, glob, shutil, mimetypes
from flask import Flask, request, jsonify, render_template, Response, abort
import torch

SEGMENT_SEC   = 15
MODEL_SIZE    = "small"
OUT_SAMPLING  = 16000
SENTINEL      = object()
ALLOWED_EXT   = {".mp3", ".wav", ".flac"}  

# Whisper ロード
try:
    from faster_whisper import WhisperModel
    ASR = WhisperModel(
        MODEL_SIZE,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "int8",
    )
    print("✓ faster-whisper loaded")
except ModuleNotFoundError:
    import whisper
    ASR = whisper.load_model(MODEL_SIZE)
    print("✓ fallback to openai/whisper")

# DLL 再生
DLL_PATH = "./stereophonic_sound_mixed/x64/Debug/stereophonic_sound_mixed.dll"
dll = ctypes.CDLL(DLL_PATH)
dll.load_audio.argtypes = [ctypes.c_char_p]

chunk_q = queue.Queue()

#  変換
def to_mono_wav(src: str) -> str:
    fd, dst = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg", "-loglevel", "quiet", "-y", "-i", src,
        "-ac", "1", "-ar", str(OUT_SAMPLING), dst
    ]
    subprocess.run(cmd, check=True)
    return dst

def split_wav(src_wav: str, seg_sec: int) -> str:
    folder = tempfile.mkdtemp()
    pattern = os.path.join(folder, "out_%03d.wav")
    cmd = [
        "ffmpeg", "-loglevel", "quiet", "-y", "-i", src_wav,
        "-f", "segment", "-segment_time", str(seg_sec),
        "-c", "copy", pattern
    ]
    subprocess.run(cmd, check=True)
    return folder

# ASR スレッド
def asr_worker(wav_folder: str):
    wav_files = sorted(glob.glob(os.path.join(wav_folder, "out_*.wav")))
    for wf in wav_files:
        if isinstance(ASR, WhisperModel):
            segs, _ = ASR.transcribe(
                wf, beam_size=1, vad_filter=False,
                temperature=0.0, word_timestamps=False)
            text = " ".join(s.text.strip() for s in segs)
        else:
            text = ASR.transcribe(wf)["text"].strip()
        if text:
            chunk_q.put(text)
    chunk_q.put(SENTINEL)
    shutil.rmtree(wav_folder, ignore_errors=True)

@app.route("/lyrics")
def page(): return render_template("lyrics.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        abort(400, "ファイルがありません。")
    f = request.files["file"]
    if not f.filename:
        abort(400, "無効なファイル名です。")

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        abort(400, f"対応していない拡張子です: {ext}")

    tmp_src = tempfile.mktemp(suffix=ext)
    f.save(tmp_src)

    # DLL にロード（再生用）
    dll.load_audio(tmp_src.encode("utf-8"))

    # キュー初期化
    while not chunk_q.empty():
        chunk_q.get_nowait()

    # 変換 → 分割 → ASR
    mono = to_mono_wav(tmp_src)
    folder = split_wav(mono, SEGMENT_SEC)
    threading.Thread(target=asr_worker, args=(folder,), daemon=True).start()

    return jsonify({"message": "認識を開始しました。"})

@app.route("/play", methods=["POST"])
def play(): threading.Thread(target=dll.play_audio, daemon=True).start() or ("", 204)

@app.route("/stop", methods=["POST"])
def stop(): dll.stop_audio() or ("", 204)

# SSE 
@app.route("/stream-lyrics")
def stream():
    def gen():
        keep = time.time()
        while True:
            try:
                txt = chunk_q.get(timeout=0.5)
                if txt is SENTINEL:
                    yield "event: done\ndata: END\n\n"
                    break
                yield f"data: {txt}\n\n"
            except queue.Empty:
                pass
            if time.time() - keep > 30:
                yield ": ping\n\n"
                keep = time.time()
    return Response(gen(), mimetype="text/event-stream")


@app.route('/song/<int:song_index>', methods=['GET'])
# song.htmlのindex→song関数のsong_index→（結果的に）song関数のindex
def song(song_index):
    # 検索対象ベクトルの取得
    query_vec = vecs_array[song_index].reshape(1, -1)
    print(f"検索対象の曲: {df.iloc[song_index]['曲名']}")

    # 検索 (類似度上位10件を取得、D：類似度スコア、I：インデックス)
    k = 11
    D, I = index_f.search(query_vec, k)
    print(f"検索結果のスコア: {D[0]}")
    print(f"検索結果のインデックス: {I[0]}")
    
    D, I = D[0][1:], I[0][1:]  # 自分自身を除外
    print(f"除外後のスコア: {D}")
    print(f"除外後のインデックス: {I}")

    # 結果を格納するリスト(<int:song_index>html、song関数下部では結果的にindex)
    results = [(df.iloc[index]['曲名'], df.iloc[index]['artist'], index, 'dummy_audio.wav') for index in I]
    save_history(dfe.iloc[song_index].to_list()) # 履歴を保存

    return render_template('song.html', song_title=df.iloc[song_index]['曲名'], results=results)

# ==========================
#  キャッシュとユーティリティ
# ==========================
_cluster_cache = {
    'means': None,
    'similarities': None,
    'timestamp': None,
    'high_correlation_features': None
}

def compute_optimal_clusters(data, max_clusters: int = 6) -> int:
    """
    エルボー法で最適クラスタ数を推定
    """
    distortions = []
    K = range(1, min(max_clusters + 1, len(data)))

    for k in K:
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    distortions = np.asarray(distortions)
    roc = (distortions[:-1] - distortions[1:]) / distortions[:-1]          # 歪みの減少率
    roc_diff = np.diff(roc)                                                # その変化率
    optimal_k = np.argmax(roc_diff) + 2                                    # +2 で調整

    return min(optimal_k, max_clusters)


# ------------------------------------------------------------
#  共通の特徴量エンジニアリング
# ------------------------------------------------------------
def process_audio_features(df: pd.DataFrame, is_display: bool = False) -> pd.DataFrame:
    """
    音声特徴量の前処理を行う
    
    Args:
        df: 入力DataFrame
        is_display: 表示用の場合はTrue（元の列を保持）
    
    Returns:
        処理済みのDataFrame
    """
    processed_df = df.copy()
    
    # 計算用の特徴量を追加（表示用では使用しない）
    if not is_display:
        processed_df["key_sin"] = np.sin(2 * np.pi * processed_df["key"] / 12)
        processed_df["key_cos"] = np.cos(2 * np.pi * processed_df["key"] / 12)
        processed_df["loudness_rel"] = processed_df["loudness"] - processed_df["loudness"].median()
        processed_df.drop(columns=["key", "loudness"], inplace=True)
    
    return processed_df


# ==========================
#  クラスタリング本体
# ==========================
def cluster_search():
    """
    - 履歴データを "計算用" と "表示用" に複製
    - 計算用 DF は追加特徴量を入れ、key/loudness を削ったうえで
      StandardScaler → 相関削除 → PCA → k-means
    - 表示用 DF は列をいじらず、計算で得た cluster ラベルだけ付与
    """
    global _cluster_cache
    now = time.time()

    # ---------- キャッシュ確認 ----------
    if (
        _cluster_cache.get("means") is not None
        and _cluster_cache.get("similarities") is not None
        and _cluster_cache.get("timestamp") is not None
        and now - _cluster_cache["timestamp"] < 300
    ):
        print("キャッシュされたクラスタリング結果を使用")
        return (
            _cluster_cache["means"],
            _cluster_cache["similarities"],
            _cluster_cache["high_correlation_features"],
        )

    print("=== cluster_search開始 ===")

    # ---------- 履歴読み込み ----------
    try:
        with open(HISTORY_FILE, newline="", encoding="utf-8") as fh:
            history_rows = list(csv.reader(fh))
    except FileNotFoundError:
        print(f"履歴ファイルが見つかりません: {HISTORY_FILE}")
        return pd.DataFrame(), [], []

    if len(history_rows) < 2:
        print("履歴データが不足しています")
        return pd.DataFrame(), [], []

    # ---------- DataFrame 基本形 ----------
    numeric_rows = [
        [float(x) if x.strip() else np.nan for x in row[1:]]
        for row in history_rows
        if len(row) > 1
    ]
    if not numeric_rows:
        print("有効な数値データがありません")
        return pd.DataFrame(), [], []

    base_df = pd.DataFrame(numeric_rows, columns=dfe.columns).dropna()
    if len(base_df) < 2:
        print("クラスタリングに十分なデータがありません")
        return pd.DataFrame(), [], []

    # ---------- ① 計算用 / ② 表示用 に複製 ----------
    calc_df = process_audio_features(base_df, is_display=False)
    disp_df = process_audio_features(base_df, is_display=True)

    # ----- 楽曲カタログ側も計算仕様を合わせる -----
    dfe_calc = process_audio_features(dfe, is_display=False)

    # ---------- スケーリング ----------
    scaler = StandardScaler()
    scaled_calc = pd.DataFrame(
        scaler.fit_transform(calc_df),
        columns=calc_df.columns,
        index=calc_df.index,
    )

    # ---------- 高相関列除去 ----------
    corr = scaled_calc.corr()
    high_corr, remove_cols = [], set()
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            c1, c2 = corr.columns[i], corr.columns[j]
            r = corr.iloc[i, j]
            if abs(r) > 0.8:
                high_corr.append((c1, c2, f"{r:.3f}"))
                remove_cols.add(c1 if scaled_calc[c1].var() < scaled_calc[c2].var() else c2)

    keep_cols = [c for c in scaled_calc.columns if c not in remove_cols]
    reduced_scaled = scaled_calc[keep_cols]

    # ---------- PCA → k-means ----------
    pca = PCA(n_components=0.95, random_state=0)
    reduced_feat = pca.fit_transform(reduced_scaled)
    k_opt = compute_optimal_clusters(reduced_feat)
    kmeans = KMeans(n_clusters=k_opt, init="k-means++", random_state=0)
    clusters = kmeans.fit_predict(reduced_feat)
    print(f"PCA {reduced_feat.shape} → k-means (k={k_opt}) 完了")

    # ---------- クラスタ平均 ----------
    # 計算用（追加列あり）
    cluster_means_calc = (
        calc_df.assign(cluster=clusters)
            .groupby("cluster").mean(numeric_only=True)
    )
    # 表示用（追加列なし）
    cluster_means_disp = (
        disp_df.assign(cluster=clusters)
            .groupby("cluster").mean(numeric_only=True)
    )
    cluster_means_disp.index.name = None

    # 'length' を秒に変換（表示用だけ）
    if "length" in cluster_means_disp.columns:
        cluster_means_disp["length"] = cluster_means_disp["length"] / 1000

    # ---------- 類似曲 (追加特徴量を含む calc 側で計算) ----------
    cluster_similarities = []
    dfe_feat_mat = dfe_calc[keep_cols].values
    for centroid in cluster_means_calc[keep_cols].values:   # ★ ここを calc に
        sims = cosine_similarity(dfe_feat_mat, centroid.reshape(1, -1)).flatten()
        top_idx = np.argsort(sims)[-10:][::-1]
        cluster_similarities.append(
            [
                (df.iloc[i]["曲名"], df.iloc[i]["artist"], int(i), "dummy_audio.wav")
                for i in top_idx
            ]
        )

    # ---------- キャッシュ ----------
    _cluster_cache.update(
        {
            "means": cluster_means_disp,          # ← 表示用 DF を保存
            "similarities": cluster_similarities,
            "timestamp": now,
            "high_correlation_features": high_corr,
        }
    )

    return cluster_means_disp, cluster_similarities, high_corr


# ==========================
#  Flask ルート
# ==========================
@app.route("/clusters", methods=["GET"])
def clusters():
    print("=== clustersルート開始 ===")
    means, sims, high_corr = cluster_search()
    print(
        f"cluster_search結果: cluster_means={len(means)}, "
        f"cluster_similarities={len(sims)}, high_correlation_features={len(high_corr)}"
    )

    if len(means) == 0 or len(sims) == 0:
        return render_template("clusters.html", message="履歴がありません")

    print("=== clustersルート終了 ===")
    return render_template(
        "clusters.html",
        cluster_means=means,
        cluster_similarities=sims,
        high_correlation_features=high_corr,
        column_names=means.columns,
    )


@app.route("/cluster_n/<int:cluster_id>")
def cluster_n(cluster_id):
    means, sims, _ = cluster_search()
    if len(sims) == 0:
        return render_template("clusters.html", message="履歴がありません")

    return render_template(
        "cluster_n.html",
        similarities=sims[cluster_id],
        link_name=f"旋律の環 {cluster_id + 1}",
    )


# ────────────────────────────────
# ここから時系列リアルタイム推薦
# ────────────────────────────────
from flask import Flask, render_template, url_for
from collections import deque
import torch, torch.nn as nn
import numpy as np
import faiss
import pickle

# --- データと設定 ---
window_size = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- モデルとスケーラーの読み込み ---
def load_model_and_scaler():
    # スケーラーの読み込み
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # モデルの読み込み
    model = LSTM(dim=len(scaler.get_feature_names_out())).to(device)
    model.load_state_dict(torch.load('lstm_model.pth', map_location=device))
    model.eval()
    
    return model, scaler

# --- 履歴の管理 ---
history = deque(maxlen=window_size)

# 履歴ファイルから履歴を読み込む
def load_history():
    global history
    try:
        with open(HISTORY_FILE, 'r', newline='') as file:
            reader = csv.reader(file)
            history_rows = list(reader)
            if len(history_rows) >= window_size:
                # 最新のwindow_size件の履歴を読み込む
                for row in history_rows[-window_size:]:
                    # タイムスタンプと特徴量を取得
                    timestamp = pd.to_datetime(row[0])
                    features = [float(x) for x in row[1:]]
                    # 特徴量をDataFrameに変換
                    features_df = pd.DataFrame([features], columns=dfe.columns)
                    # 時間特徴量を追加
                    features_df['hour'] = timestamp.hour
                    features_df['day_of_week'] = timestamp.weekday()
                    # 前処理を適用
                    processed_features = process_audio_features(features_df, is_display=False)
                    history.append(processed_features.values[0])
    except FileNotFoundError:
        print("履歴ファイルが見つかりません")
        history.clear()

# --- 再生時の部分微調整 ---
def on_play(track_features):
    """
    曲が再生されたときに呼び出される関数
    最新1000件の履歴データを使用してモデルを微調整する
    
    Args:
        track_features: 再生された曲の特徴量
    """
    global model, history
    
    # モデルとスケーラーを読み込み
    model, scaler = load_model_and_scaler()
    
    # 現在時刻を取得
    current_time = pd.Timestamp.now()
    
    # 特徴量をDataFrameに変換
    features_df = pd.DataFrame([track_features], columns=dfe.columns)
    
    # train_and_evaluate.pyと同じ順序で処理
    # 1. 特徴量エンジニアリングを適用
    processed_df = process_audio_features(features_df, is_display=False)
    
    # 2. 時間特徴量の追加
    processed_df['hour'] = current_time.hour
    processed_df['day_of_week'] = current_time.weekday()
    
    # 特徴量を正規化
    normalized_features = scaler.transform(processed_df)
    
    # 履歴に追加
    history.append(normalized_features[0])
    
    if len(history) < window_size:
        return
    
    # 履歴全体を読み込む
    try:
        with open(HISTORY_FILE, 'r', newline='') as file:
            reader = csv.reader(file)
            history_rows = list(reader)
            
            # 最新1000件の履歴データを取得
            recent_rows = history_rows[-1000:] if len(history_rows) > 1000 else history_rows
            
            # 履歴データをDataFrameに変換
            history_data = []
            for row in recent_rows[1:]:  # ヘッダーをスキップ
                if len(row) > 1:
                    try:
                        timestamp = pd.to_datetime(row[0])
                        features = [float(x) for x in row[1:]]
                        features_df = pd.DataFrame([features], columns=dfe.columns)
                        
                        # train_and_evaluate.pyと同じ順序で処理
                        # 1. 特徴量エンジニアリングを適用
                        processed_features = process_audio_features(features_df, is_display=False)
                        
                        # 2. 時間特徴量の追加
                        processed_features['hour'] = timestamp.hour
                        processed_features['day_of_week'] = timestamp.weekday()
                        
                        history_data.append(processed_features.values[0])
                    except (ValueError, TypeError):
                        continue
            
            if len(history_data) < window_size:
                return
            
            # 履歴データを正規化
            history_data = scaler.transform(history_data)
            
            # シーケンスを作成
            X, y = [], []
            for i in range(len(history_data) - window_size):
                X.append(history_data[i:i + window_size])
                y.append(history_data[i + window_size])
            
            X = torch.tensor(X, device=device, dtype=torch.float32)
            y = torch.tensor(y, device=device, dtype=torch.float32)
            
            # 学習モードに切り替え
            model.train()
            
            # 最適化器の設定
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # エポック数は少なめに設定（微調整なので）
            for epoch in range(5):
                optimizer.zero_grad()
                output = model(X)
                loss = nn.functional.mse_loss(output, y)
                loss.backward()
                optimizer.step()
            
            # モデルを保存
            torch.save(model.state_dict(), 'lstm_model.pth')
            
            # 評価モードに戻す
            model.eval()
            
    except FileNotFoundError:
        print("履歴ファイルが見つかりません")

# --- 推論: 各ステップ最適曲を1曲ずつ ---
def predict_next_songs(num_predictions=10):
    """
    次のnum_predictions曲を予測する
    
    Args:
        num_predictions: 予測する曲の数
    
    Returns:
        予測された曲の特徴量のリスト（時間特徴量を含む）と比較用特徴量（時間特徴量を除外）
    """
    global model, history
    
    if len(history) < window_size:
        return [], []
    
    # モデルとスケーラーを読み込み
    model, scaler = load_model_and_scaler()
    
    predictions = []
    # 履歴をnumpy配列に変換してからfloat32型のテンソルに変換
    history_array = np.array(list(history), dtype=np.float32)
    current_sequence = torch.tensor(history_array, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(num_predictions):
            # 予測
            pred = model(current_sequence)
            predictions.append(pred[0].cpu().numpy())
            
            # シーケンスを更新
            current_sequence = torch.cat([
                current_sequence[:, 1:],
                pred.unsqueeze(1)
            ], dim=1)
    
    # 予測結果をDataFrameに変換
    pred_df = pd.DataFrame(predictions, columns=scaler.get_feature_names_out())
    
    # 現在時刻を取得して時間特徴量を更新
    current_time = pd.Timestamp.now()
    pred_df['hour'] = current_time.hour
    pred_df['day_of_week'] = current_time.weekday()
    
    # 予測用の特徴量（時間特徴量を含む）
    prediction_features = pred_df.values
    
    # 曲データベースとの比較用の特徴量（時間特徴量を除外）
    comparison_features = pred_df.drop(columns=['hour', 'day_of_week']).values
    
    return prediction_features, comparison_features

# --- 初期化 ---
model, scaler = load_model_and_scaler()

@app.route('/timeseries')
def timeseries():
    # 履歴が不足している場合は履歴を再読み込み
    if len(history) < window_size:
        load_history()
    
    pred_features, comp_features = predict_next_songs(window_size)
    if len(pred_features) == 0:
        return render_template('timeseries.html', message='履歴が不足しています')
    
    # dfeの値を前処理
    dfe_processed = process_audio_features(pd.DataFrame(dfe.values, columns=dfe.columns), is_display=False)
    
    # 予測結果を曲情報に変換
    recommendations = []
    used_indices = set()  # 既に使用した曲のインデックスを記録
    
    for comp_feature in comp_features:
        # 予測された特徴量と最も類似する曲を検索
        similarities = cosine_similarity([comp_feature], dfe_processed.values)[0]
        
        # 類似度の高い順にインデックスを取得
        sorted_indices = np.argsort(similarities)[::-1]
        
        # まだ使用していない曲を探す
        selected_idx = None
        for idx in sorted_indices:
            if idx not in used_indices:
                selected_idx = idx
                used_indices.add(idx)
                break
        
        if selected_idx is not None:
            # 曲情報を追加
            recommendations.append((
                df.iloc[selected_idx]['曲名'],
                df.iloc[selected_idx]['artist'],
                selected_idx,
                'dummy_audio.wav'
            ))
    
    return render_template('timeseries.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
