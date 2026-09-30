# Model files go here

Copy these two files from your Phase 4 checkpoints directory into this
folder before starting the API:

```
mvit_checkpoints_gru_top30_baseline_label_vocab_best.pt
mvit_checkpoints_gru_top30_baseline_label_vocab_label_vocab.json
```

(Or whichever backbone/tag combination you trained -- the exact filenames
must match INCLUSIVE_BACKBONE and INCLUSIVE_TAG in app/config.py, or the
environment variables of the same name.)

This folder is also where pretrained-backbone downloads (VideoMAE/MViT
Kinetics-400 weights) get cached on first run, under models/cache/.
