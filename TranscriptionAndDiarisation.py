from faster_whisper import WhisperModel
import torch
import torchaudio
import numpy as np
import os
import glob
import subprocess
from speechbrain.inference import EncoderClassifier
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import shutil
import audonnx
import soundfile as sf
import audinterface
import yaml

# Charger le fichier config.yaml
with open("transcription_settings.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Configuration générale ---
audio_path = config["audio_path"]
language = config["language"]
embedding_dir = config["embedding_dir"]
modelWhisp = config["modelWhisp"]
embedding_duration = config["embedding_duration"]
transcription_duration = config["transcription_duration"]
overlap = config["overlap"]
affichageTimestampWord = config["affichageTimestampWord"]
n_speakers = config["n_speakers"]
PeopleInformation = config["PeopleInformation"]

# --- Configuration clustering ---
clustering_metric = config["clustering_metric"]
dbscan_min_samples = config["dbscan_min_samples"]
tsne_perplexity = config["tsne_perplexity"]
show_Graphique = config["show_Graphique"]

#COnfiguration classifier :
if PeopleInformation:
    model = audonnx.load('ClassificationModel')
    outputs = ['logits_age', 'logits_gender']
    interface = audinterface.Feature(
        model.labels(outputs),
        process_func=model,
        process_func_args={
            'outputs': outputs,
            'concat': True,
        },
        sampling_rate=16000,
        resample=True,
        verbose=False,
    )

# --- Fonction clustering avec DBSCAN dynamique ---
def run_clustering_and_return_labels(embeddings):
    if len(embeddings) < 5:
        return [-1] * len(embeddings), 0

    X = np.array(embeddings)
    best_score = -1
    best_labels = None
    best_eps = 0

    for test_eps in np.arange(0.1, 0.5, 0.05):
        dbscan = DBSCAN(eps=test_eps, min_samples=dbscan_min_samples, metric=clustering_metric)
        labels = dbscan.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # if n_clusters < 1:
        #     continue

        try:
            score = silhouette_score(X, labels, metric=clustering_metric)
            print (f"score pour eps {test_eps} : ", score)
        except Exception:
            continue

        if score > best_score:
            best_score = score
            best_labels = labels
            best_eps = test_eps

    if best_labels is None:
        return [-1] * len(X), 0

    if show_Graphique:
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
        reduced = tsne.fit_transform(X)
        plt.figure(figsize=(10, 8))
        unique_labels = set(best_labels)
        colors = plt.cm.get_cmap('tab10', len(unique_labels))
        for label in unique_labels:
            indices = np.where(np.array(best_labels) == label)[0]
            color = 'k' if label == -1 else colors(label % 10)
            plt.scatter(reduced[indices, 0], reduced[indices, 1],
                        label=f"Locuteur {label + 1}" if label != -1 else "Inconnu", c=[color])
        plt.title(f"Clustering des locuteurs (eps : {best_eps:.2f})")
        plt.legend()
        plt.grid(True)
        plt.show()

    return best_labels, len(set(best_labels) - {-1})


def run_kmeans_clustering(embeddings, n_clusters=2, show_graph=True):
    if len(embeddings) < n_clusters:
        return [-1] * len(embeddings), 0

    X = np.array(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)

    if show_graph:
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
        reduced = tsne.fit_transform(X)
        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap('tab10', n_clusters)
        for label in range(n_clusters):
            indices = np.where(np.array(labels) == label)[0]
            plt.scatter(reduced[indices, 0], reduced[indices, 1],
                        label=f"Locuteur {label}", c=[colors(label % 10)])
        plt.title(f"Clustering KMeans (k = {n_clusters})")
        plt.legend()
        plt.grid(True)
        plt.show()

    return labels, n_clusters


device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"
whisper_model = WhisperModel(modelWhisp, device=device, compute_type=compute_type)
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)

# Découpage pour clustering
def split_audio_with_ffmpeg(input_file, output_dir, segment_duration):
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "ffmpeg",
        "-i", input_file,
        "-f", "segment",
        "-segment_time", str(segment_duration),
        "-c", "copy",
        os.path.join(output_dir, f"segment_%03d.wav"),
        "-loglevel", "error"
    ]
    subprocess.run(command)

split_audio_with_ffmpeg(audio_path, embedding_dir, embedding_duration)


embedding_files = sorted(glob.glob(os.path.join(embedding_dir, "*.wav")))
embedding_segments = []

for idx, file in enumerate(embedding_files):


    waveform, sr = torchaudio.load(file)

    # if waveform.abs().mean() < 0.01:
    #     print("segment silencieux")
    #     continue

    embedding = classifier.encode_batch(waveform.to(device))
    embedding_np = embedding.squeeze().cpu().numpy()

    if embedding_np.ndim > 1:
        embedding_np = embedding_np.reshape(-1)
    embedding_segments.append({
        "path": file,
        "embedding": embedding_np,
        "start_time": idx * embedding_duration
    })

# #Clustering dynamique
# X = [s["embedding"] for s in embedding_segments]
# labels, n_speakers = run_clustering_and_return_labels(X)

# Clustering manuel
X = [s["embedding"] for s in embedding_segments]
labels, n_speakers = run_kmeans_clustering(X, n_clusters=n_speakers)
#print("label :" , labels)

# import matplotlib.pyplot as plt
#
# norms = [np.linalg.norm(e) for e in X]
# plt.hist(norms, bins=20)
# plt.title("Distribution des normes des embeddings")
# plt.show()

for segment, label in zip(embedding_segments, labels):
    segment["speaker"] = label
    petitbout = segment["path"]

    signal, sr = sf.read(petitbout, dtype='float32')

    if signal.ndim > 1:
        signal = signal.mean(axis=1)  # mono

    if PeopleInformation:
        result = interface.process_signal(signal, sr)

        #prédictions les plus probables
        predicted_gender = result.drop(columns=['age']).idxmax(axis=1).values[0]
        predicted_age = result['age'].values[0]

        segment["predicted_gender"] = predicted_gender
        segment["predicted_age"] = predicted_age



#Transcription avec entrelacement
waveform, sr = torchaudio.load(audio_path)
total_duration = waveform.shape[1] / sr

transcription_segments = []
start = 0
while start < total_duration:
    end = min(start + transcription_duration, total_duration)
    segment_waveform = waveform[:, int(start * sr):int(end * sr)]

    segment_path = f"segment_{int(start)}s.wav"
    torchaudio.save(segment_path, segment_waveform, sr)
    transcription_segments.append((segment_path, start, end))

    start += transcription_duration - overlap

results = []

print("\n--- Résultat final ---\n")
locuteurSave = -2
for file, start_time, end_time in transcription_segments:
    segments, _ = whisper_model.transcribe(
        file,
        language=language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True
    )

    wordAffichage = []

    IsLocuteurDifferent = False
    for seg in segments:
        for word in seg.words:
            global_start_time = start_time + word.start

            locuteur = "Inconnu"
            for emb in embedding_segments:
                #print("speaker = ",emb["speaker"], "time start : ",emb["start_time"])
                seg_start = emb["start_time"]
                seg_end = seg_start + embedding_duration
                if seg_start <= global_start_time < seg_end:
                    locuteur = f"Locuteur {emb['speaker']}" if emb["speaker"] != -1 else "Inconnu"
                    if(locuteurSave != emb["speaker"]):
                        IsLocuteurDifferent = True
                    else:
                        IsLocuteurDifferent = False

                    locuteurSave = emb["speaker"]
                    break


            if affichageTimestampWord and IsLocuteurDifferent:
                wordAffichage.append(f"({global_start_time:.2f}s, {locuteur}) {word.word}")
            else:
                wordAffichage.append(word.word)




    full_text = " ".join(wordAffichage)

    speakers_in_segment = set()
    for segment in embedding_segments:
        if start_time <= segment["start_time"] < end_time and segment["speaker"] != -1:
            speakers_in_segment.add(segment["speaker"])

    results.append({
        "start": start_time,
        "end": end_time,
        "speakers": sorted(speakers_in_segment),
        "transcription": full_text
    })

    os.remove(file)

if PeopleInformation:

    from collections import defaultdict

    speaker_info = defaultdict(lambda: {"ages": [], "genders": []})

    for seg in embedding_segments:
        speaker = seg["speaker"]
        if speaker != -1:
            speaker_info[speaker]["ages"].append(seg["predicted_age"])
            speaker_info[speaker]["genders"].append(seg["predicted_gender"])

    # Affichage des moyennes par locuteur
    print("\n--- Caractéristiques par locuteur ---")
    for speaker, info in speaker_info.items():
        age_moyen = np.mean(info["ages"]) * 100
        genres = info["genders"]
        genre_majoritaire = max(set(genres), key=genres.count)

        print(f"Locuteur {speaker} : âge moyen ≈ {age_moyen:.1f}, genre : {genre_majoritaire}")

#Affichage
for r in results:
    start = int(r["start"])
    end = int(r["end"])
    speakers_str = ", ".join([f"Locuteur {s}" for s in r["speakers"]])
    print(f"[{start}s - {end}s] : {r['transcription']}")

#Exportation
with open(f"{audio_path}_transcription.txt", "w", encoding="utf-8") as f:
    if PeopleInformation:
        for speaker, info in speaker_info.items():
            age_moyen = np.mean(info["ages"]) * 100
            genres = info["genders"]
            genre_majoritaire = max(set(genres), key=genres.count)
            f.write(f"Speaker {speaker} — Average Age: {age_moyen:.1f}, Gender: {genre_majoritaire}\n")

    f.write("\n--- Transcription ---\n")
    for r in results:
        start = int(r["start"])
        end = int(r["end"])
        speakers_str = ", ".join([f"Speaker {s}" for s in r["speakers"]])
        f.write(f"[{start}s - {end}s] ({speakers_str}) : {r['transcription']}\n")

    print("Transcription complete!")

# Suppression des fichiers temporaires dans embedding_dir à la fin du script
shutil.rmtree(embedding_dir)
