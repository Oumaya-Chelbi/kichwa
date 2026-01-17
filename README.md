Ce dépôt contient le code et les données associés à une expérience de reconnaissance automatique de la parole (ASR) pour le kichwa, avec un focus spécifique sur le phénomène de code-switching entre l’espagnol et le kichwa.
L’objectif principal est d’évaluer dans quelle mesure le fine-tuning d’un modèle Whisper sur un corpus monolingue en kichwa influence la reconnaissance de séquences mixtes espagnol–kichwa, par rapport au comportement du modèle de base.

## Données

Le dossier `data/` contient l’ensemble des ressources utilisées pour nos expériences :

- les différents corpus audio + texte préparés pour l’ASR ;
- le fichier d’annotations  au format CoNLL-U ;
- le fichier `codeswitch.tsv` indiquant quels segments contiennent du code-switching.

### Organisation du dossier `data/`

- `data/CORPUS_qug/`  
  Corpus normalisé dérivé de Killkan, avec un répertoire plat contenant des paires `wav/txt` et un fichier `corpus_metadata.tsv` décrivant la provenance (chapitre, locuteur, etc.).

- `data/CORPUS_qug_Cswitch/`  
  Sous-corpus contenant uniquement les segments identifiés comme contenant du code-switching (espagnol–kichwa).

- `data/CORPUS_qug_NO_Cswitch/`  
  Sous-corpus complémentaire, ne contenant que des segments monolingues (sans code-switching).

- `data/CORPUS_qug_NO_Cswitch_30/`  
  Version filtrée de `CORPUS_qug_NO_Cswitch/` ne gardant que les audios de durée ≤ 30 secondes, utilisée pour le fine-tuning de Whisper.

- `data/kc_killkan-ud-test.conllu`  
  Fichier CoNLL-U fourni avec Killkan, contenant les segments alignés avec leur traduction, les timecodes et les annotations morphosyntaxiques (tags UD, champ `Lang`, etc.).

- `data/codeswitch.tsv`  
  Fichier tabulé construit à partir du CoNLL-U, listant pour chaque fichier audio une étiquette `has_codeswitch` (0/1) indiquant la présence de code-switching dans le segment.

## Scripts

Le dossier `scripts/` contient l’ensemble des scripts utilisés pour préparer les données et entraîner/évaluer les modèles.

### Préparation du corpus ASR

- `mp4_to_wav.py`  
  Parcourt récursivement l’arborescence Killkan et convertit tous les fichiers `.mp4` en `.wav` mono 16 kHz, en conservant la structure de dossiers et les noms de base.

- `eaf_to_txt.py`  
  Extrait les transcriptions des fichiers ELAN (`.eaf`), applique un léger nettoyage (balises, espaces) et crée un fichier `.txt` par enregistrement audio.

- `build_corpus.py`  
  Reconstruit un corpus « plat » à partir de l’arborescence initiale : renomme les fichiers en encodant leur provenance (chapitre, section, etc.) et crée le répertoire `CORPUS_qug/` avec les paires `wav/txt` et un `corpus_metadata.tsv`.

- `detect_codeswitch.py`  
  Analyse le fichier `kc_killkan-ud-test.conllu` pour détecter les segments où plusieurs langues coexistent et génère le fichier `codeswitch.tsv` indiquant, pour chaque fichier audio, la présence (`1`) ou non (`0`) de code-switching.

- `copy_codeswitch_audio.py`  
  À partir de `codeswitch.tsv`, copie dans `CORPUS_qug_Cswitch/` toutes les paires `wav/txt` contenant du code-switching.

- `copy_non_codeswitch_audio_and_txt.py`  
  Copie de manière complémentaire les paires sans code-switching dans `CORPUS_qug_NO_Cswitch/`.

- `30s.py`  
  Filtre `CORPUS_qug_NO_Cswitch/` pour ne garder que les enregistrements de durée ≤ 30 secondes et construit `CORPUS_qug_NO_Cswitch_30/`, utilisé pour le fine-tuning.

### Entraînement et inférence Whisper

- `ft_whisper.py`  
  Contient la logique du fine-tuning de `whisper-base` (préparation des données, module Lightning, configuration du `Trainer`, calcul du WER/CER en validation).

- `real_ft_whisper.py`  
  Script "d'execution" qui charge les fonctions de `ft_whisper.py`, lance l’entraînement sur `CORPUS_qug_NO_Cswitch_30/`, sauvegarde le meilleur modèle et calcule les scores finaux sur le corpus de test.(nb : ce fichier est bloqué je n'arrive pas à le push si besoin je poeux toujours l'envoyer par mail)

- `bon_whisper.py`  
  Script d’inférence qui applique soit le modèle Whisper de base, soit le modèle fine-tuné, à un corpus donné et enregistre les transcriptions dans un fichier `.tsv` (colonnes `file`, `transcription`, `duration`).

NB : Le script d'évaluation ce trouve dans l'autre github 

## SAVE_DIR :
Contient les poids de nôtre modèle fine-tuner pour pouvoir le réutilisé par la suite

## Results : 
Le dossier `results/` contient l’ensemble des résultats obtenus :

`evals/` : contient les résultats wer et cer de toutes les transcriptions

`images/`: contient les images utilisées dans le rapport

`transcription/`: contient toutes les transcriptions obtenues (nb : whisper_base_qug_codeswitch_es et whisper_base_qug_no_codeswitch_es ont été obtenue en forçant le modèle de base à prendre l'espagnole comme langue de transcription mais nous avons utilisé celle où la détéction se fait automatiquement comme baseline en raison de leurs meilleurs score wer )

## Dump : 
Continent tous les fichiers que l'on a pas utilisées finalement

## Autre : 
`fichier_log_entraînement`: fichier de la sortie terminale lorsque nous avons lander le fine-tuning 

`requirements.txt`: Liste des dépendances Python nécessaires.
