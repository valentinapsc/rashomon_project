# Rashomon su MNIST 

Questo progetto analizza l’effetto **Rashomon** su MNIST costruendo un set di modelli
equivalenti in accuratezza e valutando le spiegazioni locali con tre metodi XAI: Saliency, Integrated Gradients (IG) e LIME. La similarità è misurata
(con **media ± dev. std**) tramite SSIM, Pearson, Spearman, Cosine, MAE;
la fedeltà è valutata con MoRF/AOPC.

---

## Contenuti principali

- **Rashomon set**: addestramento di più CNN su MNIST; selezione dei modelli entro **±1%**
  dalla miglior accuracy di validazione.
- **Metodi XAI**: Saliency, Integrated Gradients, LIME.
- **Similarità**: intra-modello (coppie di metodi sullo stesso modello) e inter-modello
  (stesso metodo su modelli diversi) con metriche: SSIM, Pearson, Spearman, Cosine, MAE.
- **Fedeltà**: MoRF & AOPC.
- **Documentazione**: report LaTeX/PDF con figure, tabelle e discussione del trade-off
  tra stabilità e fedeltà.

---

## Requisiti

- Python 3.9–3.11
- Pacchetti principali: `torch`, `torchvision`, `captum`, `numpy`, `scipy`,
  `scikit-image`, `matplotlib`, `tqdm`, `pillow`

Possibile installare con `requirements.txt`:

```bash
pip install -r requirements.txt
```
## Come eseguire
```bash
python main.py
```

