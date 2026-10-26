# Données et méthode

## Sources publiques

| Rôle | Jeu de données | Contenu |
| --- | --- | --- |
| Profils tumoraux | TCGA via le portail GDC (accès ouvert) | 450K, 8 classes × 48 tumeurs, un patient par tumeur |
| Fond sanguin | Hannum et al. 2013, GEO GSE40279 | Sang total de 164 adultes, 450K |
| Plasma réel | Moss et al. 2018, GEO GSE122126 | ADN circulant (EPIC) : 12 sains, 4 poumon, 4 colorectal, 3 sein, 4 primitif inconnu |

Classes d'organe d'origine : poumon, colorectal, sein, pancréas, foie, œsophage-estomac,
tête et cou, prostate.

## Découpage

Individus sains et tumeurs sont séparés en entraînement (70 %) et test (30 %) **avant** toute
simulation et avant le choix du panel. Aucune personne ni tumeur du test n'est vue à
l'entraînement. Le plasma réel de Moss et al. ne sert qu'à la validation finale.

## Panel ciblé

1 692 sites CpG retenus parce qu'ils sont quasi non méthylés dans le sang (moyenne < 0,10,
95e centile < 0,20) et méthylés dans au moins un type tumoral. Le bruit de fond du plasma y est
faible, donc quelques fragments tumoraux suffisent à les faire ressortir.

## Simulation d'un plasma

1. On tire un individu sain réel : c'est le fond, l'ADN libéré par les cellules sanguines.
2. Pour un cas de cancer, on mélange une tumeur réelle de la classe voulue à une fraction
   tumorale de 0,01 % à 10 %.
3. On simule le séquençage : profondeur variable par CpG (log-normale autour de 500 lectures),
   lectures méthylées tirées selon une loi binomiale.

**Limite assumée** : la simulation se fait CpG par CpG, sans motifs au niveau de la lecture.
Les tests commerciaux exploitent ces motifs et sont plus sensibles aux très faibles fractions
tumorales.

## Reproduire

```bash
make data reference demo-samples
```
