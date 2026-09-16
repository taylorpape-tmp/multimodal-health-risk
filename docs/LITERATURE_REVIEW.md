# Literature review: multimodal / multi-omics machine learning for diabetes and insulin-resistance prediction

Prepared to benchmark the HURDLE pipeline's methodological choices against the published field and to supply
verified citations for the project report. Every paper below was retrieved from a live scholarly database
(PubMed/PMC via the NCBI connector; CrossRef and arXiv for computer-science method papers). DOIs are copied
from the retrieved records. OpenAlex was requested but the credential was declined for this session, so it was
not used; no citation here depends on it.

HURDLE's methodological stance, for reference: **late (decision-level) fusion** across four modalities
demonstrated on a latent-`z` virtual cohort; **XGBoost on tabular omics** under leave-one-out CV with a
PCA/variance/correlation reduction funnel; **hand-crafted CGM variability features** (time-in-range, mean
glucose, glucotype descriptors); **cosinor/circadian + HRV wearable features**; a **transfer-learning retinal
CNN** (RETFound / ResNet-50 / DINOv2 backbones); **nested-consensus feature selection with SHAP** for
interpretability; and **permutation + row-scramble negative controls** for validation, plus a
differential-privacy / federated-averaging layer.

---

## 1. Multimodal fusion for T2D risk

The field's organising vocabulary, early (input/feature-concatenation) fusion, intermediate (joint
latent) fusion, and late (decision-level) fusion, is set out in two widely-used reviews. [Huang et al. 2020](https://doi.org/10.1038/s41746-020-00341-z)
systematically reviewed fusion of medical imaging with electronic health records and codified the
early/joint/late taxonomy together with implementation guidance; [Stahlschmidt et al. 2022](https://doi.org/10.1093/bib/bbab569) reviewed multimodal deep
learning for biomedical data fusion specifically and reached the recurring conclusion that no single fusion
stage dominates, the right choice is dictated by how tightly the modalities are coupled and how much aligned
per-patient data exists. [Lipkova et al. 2022](https://doi.org/10.1016/j.ccell.2022.09.012) made the same argument for oncology, where late/decision-level
fusion is the pragmatic default when modalities are only partially co-observed. On the diabetes side,
multimodal deep learning is now common: [Ding et al. 2024](https://doi.org/10.1038/s41598-024-71020-2) used large multimodal models over five-year EHR
cohorts for new-onset T2D prediction, and [Haleem et al. 2025](https://doi.org/10.1038/s41598-025-07272-3) built a multimodal deep architecture for
interstitial-glucose prediction, both fuse heterogeneous streams, but both also enjoy a single linked cohort,
which is precisely the resource HURDLE lacks.

**How HURDLE compares.** HURDLE's choice of **late fusion** is squarely within field norms and is the
*correct* choice for its data situation: the four public datasets are different patients, so an early-fusion
concatenation or a jointly-trained intermediate encoder, which need per-patient aligned rows, is simply not
trainable on the real data. Late fusion combines per-modality predictions and tolerates missing/disjoint
modalities, which is exactly why the reviews above name it the default under weak coupling. Where HURDLE is
*simpler* than the DL literature is that its fusion is demonstrated on a virtual cohort rather than a real
linked one; where it is *more rigorous* than most is the explicit scramble control (Topic 8). This is a
reasonable, well-motivated deviation given student-scale, unlinked data.

## 2. Multi-omics integration methods for metabolic disease

Three method families dominate. **Latent-factor models**: [Argelaguet et al. 2018](https://doi.org/10.15252/msb.20178124) introduced Multi-Omics Factor
Analysis (MOFA), an unsupervised framework that learns a shared low-dimensional factor space across omics
layers; it is the reference method for unsupervised metabolic multi-omics. **Supervised
latent-component/PLS**: [Singh et al. 2019](https://doi.org/10.1093/bioinformatics/bty1054) introduced DIABLO within mixOmics, a supervised approach that finds
correlated multi-omics components discriminating known groups and returns interpretable molecular drivers.
**Similarity/graph fusion**: [Wang et al. 2014](https://doi.org/10.1038/nmeth.2810) introduced Similarity Network Fusion (SNF), which builds a
per-modality patient-similarity network and fuses them into one consensus network for subtyping. More
recently, deep generative and autoencoder integrators have appeared: [Allesøe et al. 2023](https://doi.org/10.1038/s41587-022-01520-x) used generative
deep-learning models to discover drug–omics associations *specifically in type-2-diabetes cohorts*, and
[Yao et al. 2024](https://doi.org/10.1186/s13040-024-00360-6) (MOCAT) is representative of auxiliary-classifier autoencoders that learn a joint omics
embedding. The through-line is that the field's high-end integration is *intermediate* fusion, a shared
latent space, not concatenation.

**How HURDLE compares.** HURDLE uses **concatenation + tree models** (PCA-reduced tabular omics → XGBoost)
rather than MOFA/DIABLO/SNF. This is the *standard baseline* the integration literature benchmarks against,
and it is defensible at n≈59 patients where MOFA's factor estimation and DIABLO's component selection are
data-hungry and unstable. HURDLE is therefore **simpler than the field's frontier** on this axis by design.
The honest gap: HURDLE treats omics as one already-assembled tabular block and does not perform cross-layer
integration (transcriptome × metabolome × proteome) as a modelled step, DIABLO or MOFA would be the
principled way to add that, and both are named as extensions below.

## 3. iPOP / Snyder-lab lineage

HURDLE's three real datasets come directly from the Stanford integrative-personal-omics (iPOP) programme.
[Zhou et al. 2019](https://doi.org/10.1038/s41586-019-1236-x) is the anchor: longitudinal multi-omics of host–microbe dynamics in prediabetes (106
individuals), the source of HURDLE's omics→SSPG matrix. [Hall et al. 2018](https://doi.org/10.1371/journal.pbio.2005143) defined **glucotypes**, CGM-derived
categories of glucose dysregulation, and is the source of HURDLE's CGM cohort and glucotype descriptors.
[Li et al. 2017](https://doi.org/10.1371/journal.pbio.2001402) established that wearable biosensors carry health-relevant physiological signal, the basis for
HURDLE's wearable block. The programme's downstream ML is instructive: [Dunn et al. 2021](https://doi.org/10.1038/s41591-021-01339-0) showed that wearable
vital signs can *predict clinical laboratory measurements* per-person (regression models on continuous sensor
features), and [Schüssler-Fiorenza Rose et al. 2019](https://doi.org/10.1038/s41591-019-0414-6) framed the whole enterprise as a longitudinal big-data approach to precision
health. These follow-ups largely stay within *per-modality* predictive modelling on the real linked cohort
rather than fusing all modalities into one risk score.

**How HURDLE compares.** HURDLE reproduces the iPOP single-modality analyses faithfully (omics→SSPG regression
is a direct re-implementation) and then attempts the *cross-modality* step the original programme did not fully
close in public data, which is why it must use a virtual cohort. So HURDLE is **standard** where it mirrors the
Snyder-lab per-modality models and **an honest extension** where it fuses them. Its per-modality feature
engineering (cosinor HRV, CGM variability) is consistent with what Dunn 2021 and the glucotype work extract.

## 4. CGM glucose-variability features in ML

The CGM-variability-index literature is where HURDLE's hand-crafted CGM features have their strongest
precedent. [Marling et al. 2011](https://doi.org/10.1177/193229681100500408) catalogued glucose-variability metrics (the MAGE/CONGA/MODD family) computable
from CGM traces. [Acciaroli et al. 2017](https://doi.org/10.1177/1932296817710478) then showed these GV indices can *classify* diabetes/prediabetes status, and
[Longato et al. 2019](https://doi.org/10.1177/1932296819838856) showed a **reduced** set of GV indices feeding a simple linear SVM distinguishes impaired
glucose tolerance from T2D, i.e. hand-crafted features + a linear model, not deep learning, is a published,
competitive recipe. [Zeevi et al. 2015](https://doi.org/10.1016/j.cell.2015.11.001) (Zeevi) is the landmark showing that features engineered from CGM plus
covariates predict *personalised* postprandial glycemic responses.

**How HURDLE compares.** HURDLE's **hand-crafted CGM features** (time-in-range, mean glucose, glucotype
descriptors) are **exactly what this field does**, the GV-index tradition is explicitly feature-engineering
over CGM, and small-n linear/tree models on those features are standard. HURDLE is neither simpler nor more
complex here; it is on-method. The one enrichment the field offers that HURDLE has not fully exploited is the
*breadth* of the GV-index panel (MAGE, CONGA, MODD, J-index, LBGI/HBGI); HURDLE could widen its CGM feature
set toward the Marling/Acciaroli panels at essentially zero methodological risk.

## 5. Wearable circadian and HRV features

Heart-rate-variability and circadian-rhythm features are an established input for metabolic ML.
[Fengade et al. 2025](https://doi.org/10.7759/cureus.80933) built ML models predicting T2D from HRV features, and [Aggarwal et al. 2020](https://doi.org/10.1007/s13246-020-00950-8) used time-domain HRV
features for automated diabetes prediction, both confirm HRV as a legitimate metabolic predictor.
On the circadian side, [Kim et al. 2025](https://doi.org/10.2196/69328) derived circadian biomarkers from wearable data for metabolic-syndrome
detection, and [Kim et al. 2026](https://doi.org/10.1177/20552076261458929) is a representative *interpretable* wearable study using heart-rate circadian
phase (cosinor-style amplitude/acrophase) as digital-phenotyping features. Cosinor decomposition
(MESOR/amplitude/acrophase) is the standard parameterisation these studies use.

**How HURDLE compares.** HURDLE's **cosinor fit of resting heart rate** (MESOR, amplitude, acrophase) plus HRV
features is **directly aligned** with this literature, cosinor and HRV time-domain features are the field's
staple wearable representation. HURDLE is standard here. The available enrichment is frequency-domain HRV
(LF/HF) and rest–activity/actigraphy rhythm features, which several of the wearable studies add on top of
cosinor.

## 6. Retinal imaging deep learning for diabetes

Retinal fundus DL has moved from task-specific CNNs to foundation models. [Gulshan et al. 2016](https://doi.org/10.1001/jama.2016.17216) (Gulshan) is the
landmark supervised CNN for diabetic-retinopathy detection. [Zhou et al. 2023](https://doi.org/10.1038/s41586-023-06555-x) introduced **RETFound**, a
self-supervised retinal foundation model pre-trained on ~1.6M unlabelled images that transfers to downstream
disease-detection tasks with far less labelled data, HURDLE's named backbone. Beyond DR grading, retinal DL
now predicts *systemic* outcomes: [Dai et al. 2024](https://doi.org/10.1038/s41591-023-02702-z) predicts time-to-progression of DR, and [Nusinovici et al. 2024](https://doi.org/10.1016/S2666-7568%2824%2900089-8)
derived a retinal biological-ageing marker (RetiPhenoAge) predicting morbidity/mortality, evidence that
retinal embeddings carry systemic-metabolic signal, which is the premise of wiring retinal features into a
diabetes-risk fusion.

**How HURDLE compares.** HURDLE's **transfer-learning CNN with a RETFound/ResNet-50/DINOv2 backbone and
class-weighted loss** is the current standard recipe: foundation-model-or-ImageNet transfer, macro-AUC/macro-F1
reporting under class imbalance. HURDLE is **on-method** but operating on a *proxy* dataset, RetinaMNIST
(28→224px MedMNIST tiles, [Yang et al. 2023](https://doi.org/10.1038/s41597-022-01721-8)) rather than full-resolution fundus photographs, and on a DR cohort
disjoint from its metabolic patients. That is a real limitation (resolution loss, no metabolic label on the
imaging cohort), honestly disclosed, and reasonable for a compute- and data-limited student project. RETFound
was built precisely to reduce the labelled-data burden HURDLE faces.

## 7. Interpretability and consensus selection

Two things are treated as best practice. First, **SHAP** ([Lundberg & Lee 2017](https://doi.org/10.48550/arXiv.1705.07874)) is the field-standard model-agnostic
attribution method; [Ponce-Bobadilla et al. 2024](https://doi.org/10.1111/cts.70056) is a recent practical guide to using SHAP correctly in biomedical
modelling, and interpretable-ML clinical models routinely pair a tree model with SHAP ([Guan et al. 2024](https://doi.org/10.1186/s13054-024-05138-0)).
Second, single-run feature importance is known to be unstable, so the methodological literature pushes
**consensus / stability** selection: [Meinshausen & Bühlmann 2010](https://doi.org/10.1111/j.1467-9868.2010.00740.x) (stability selection) and [Kursa & Rudnicki 2010](https://doi.org/10.18637/jss.v036.i11) (Boruta
all-relevant selection) are the canonical methods, and empirical benchmarks, [Li et al. 2022](https://doi.org/10.1186/s12859-022-04962-x) on multi-omics
feature-selection strategies and [Degenhardt et al. 2019](https://doi.org/10.1093/bib/bbx124) on variable selection for random forests on omics, show that
aggregating across resamples/methods gives more reproducible panels than any single selector.

**How HURDLE compares.** HURDLE's **nested-consensus feature selection (filter/wrapper/embedded/explain) plus
SHAP** is **at or slightly above field best practice**. Combining SHAP with a multi-strategy consensus panel is
exactly what the stability-selection and FS-benchmark literature recommends, and doing the consensus *inside*
nested CV avoids the selection-leakage that the benchmarks warn about. This is arguably the strongest-aligned
component of the whole pipeline, HURDLE is not simpler here, it is methodologically current.

## 8. Virtual cohorts and negative controls

The validation controls HURDLE uses have direct precedent. [Mi et al. 2021](https://doi.org/10.1038/s41467-021-22756-2) established **permutation-based**
identification of important biomarkers, rerunning models on permuted labels to separate real signal from
chance, and [Cai et al. 2022](https://doi.org/10.1073/pnas.2205518119) developed a formal model-free prediction (permutation) test for genomics data.
[Tran et al. 2025](https://doi.org/10.1093/molbev/msaf250) used haplotype-matrix permutations to interpret and validate supervised ML in genomics.
Permutation/negative-control testing is thus an accepted rigour standard; synthetic/virtual cohorts with a
known latent generator are a recognised way to test whether a fusion model recovers coupling that genuinely
exists.

**How HURDLE compares.** HURDLE's **label-permutation p-values (permuting outside the CV) and its row-scramble
fusion control** are **more rigorous than the median paper** in Topics 1–6, most of which report a single
cross-validated metric with no negative control. Running the permutation *outside* the CV loop is the
leakage-safe form the statistics literature calls for. The virtual-cohort design, drawing a hidden latent
risk `z` and generating each modality from it, then requiring fusion to beat both the best single modality and
a scrambled control, is a defensible way to demonstrate the fusion machinery when no real linked cohort
exists, provided (as HURDLE does) it is disclosed as a demonstration rather than a clinical result.

---

## Methods the field uses that HURDLE could add

1. **MOFA and/or DIABLO for genuine cross-omics integration.** The multi-omics literature ([Argelaguet et al. 2018](https://doi.org/10.15252/msb.20178124),
   [Singh et al. 2019](https://doi.org/10.1093/bioinformatics/bty1054)) treats a shared latent factor space (MOFA, unsupervised) or supervised correlated
   components (DIABLO/mixOmics) as the principled way to integrate transcriptome × metabolome × proteome.
   HURDLE currently concatenates a pre-assembled omics block into XGBoost; adding a MOFA/DIABLO layer on the
   real S4/S8 matrices would upgrade the omics step from "tabular baseline" to "field-standard integration"
   and would yield interpretable cross-layer factors.
2. **Intermediate (joint-latent) fusion as a benchmark against late fusion.** The fusion reviews
   ([Huang et al. 2020](https://doi.org/10.1038/s41746-020-00341-z), [Stahlschmidt et al. 2022](https://doi.org/10.1093/bib/bbab569)) present intermediate fusion as the higher-capacity option when aligned
   data exists. On a real single-cohort dataset HURDLE could run an intermediate-fusion encoder and report it
   head-to-head against its late-fusion baseline, turning the fusion-stage choice into an empirical result
   rather than a constraint.
3. **A broader validated CGM glucose-variability panel + retinal foundation embeddings.** The CGM literature
   ([Marling et al. 2011](https://doi.org/10.1177/193229681100500408), [Acciaroli et al. 2017](https://doi.org/10.1177/1932296817710478)) uses a wide GV-index panel (MAGE/CONGA/MODD/J-index/LBGI-HBGI);
   HURDLE could adopt it wholesale at low risk. In parallel, replacing the RetinaMNIST proxy with genuine
   RETFound embeddings ([Zhou et al. 2023](https://doi.org/10.1038/s41586-023-06555-x)) on full-resolution fundus images would let the imaging block enter
   fusion as a real feature vector rather than a synthetic placeholder.

---

## References

All entries were retrieved from PubMed/PMC (NCBI), CrossRef, or arXiv during this review.

1. Huang; Pareek; Seyyedi; Banerjee. Fusion of medical imaging and electronic health records using deep learning: a systematic review and implementation guidelines. *NPJ Digit Med* (2020). doi:[10.1038/s41746-020-00341-z](https://doi.org/10.1038/s41746-020-00341-z)
2. Stahlschmidt; Ulfenborg; Synnergren. Multimodal deep learning for biomedical data fusion: a review. *Brief Bioinform* (2022). doi:[10.1093/bib/bbab569](https://doi.org/10.1093/bib/bbab569)
3. Lipkova; Chen; Chen; Lu. Artificial intelligence for multimodal data integration in oncology. *Cancer Cell* (2022). doi:[10.1016/j.ccell.2022.09.012](https://doi.org/10.1016/j.ccell.2022.09.012)
4. Ding; Thao; Peng; Wang. Large language multimodal models for new-onset type 2 diabetes prediction using five-year cohort electronic health records. *Sci Rep* (2024). doi:[10.1038/s41598-024-71020-2](https://doi.org/10.1038/s41598-024-71020-2)
5. Haleem; Katsarou; Georga; Dafoulas. A multimodal deep learning architecture for predicting interstitial glucose for effective type 2 diabetes management. *Sci Rep* (2025). doi:[10.1038/s41598-025-07272-3](https://doi.org/10.1038/s41598-025-07272-3)
6. Argelaguet; Velten; Arnol; Dietrich. Multi-Omics Factor Analysis-a framework for unsupervised integration of multi-omics data sets. *Mol Syst Biol* (2018). doi:[10.15252/msb.20178124](https://doi.org/10.15252/msb.20178124)
7. Singh; Shannon; Gautier; Rohart. DIABLO: an integrative approach for identifying key molecular drivers from multi-omics assays. *Bioinformatics* (2019). doi:[10.1093/bioinformatics/bty1054](https://doi.org/10.1093/bioinformatics/bty1054)
8. Wang; Mezlini; Demir; Fiume. Similarity network fusion for aggregating data types on a genomic scale. *Nat Methods* (2014). doi:[10.1038/nmeth.2810](https://doi.org/10.1038/nmeth.2810)
9. Allesøe; Lundgaard; Hernández Medina; Aguayo-Orozco. Discovery of drug-omics associations in type 2 diabetes with generative deep-learning models. *Nat Biotechnol* (2023). doi:[10.1038/s41587-022-01520-x](https://doi.org/10.1038/s41587-022-01520-x)
10. Yao; Jiang; Luo; Liang. MOCAT: multi-omics integration with auxiliary classifiers enhanced autoencoder. *BioData Min* (2024). doi:[10.1186/s13040-024-00360-6](https://doi.org/10.1186/s13040-024-00360-6)
11. Zhou; Sailani; Contrepois; Zhou. Longitudinal multi-omics of host-microbe dynamics in prediabetes. *Nature* (2019). doi:[10.1038/s41586-019-1236-x](https://doi.org/10.1038/s41586-019-1236-x)
12. Hall; Perelman; Breschi; Limcaoco. Glucotypes reveal new patterns of glucose dysregulation. *PLoS Biol* (2018). doi:[10.1371/journal.pbio.2005143](https://doi.org/10.1371/journal.pbio.2005143)
13. Li; Dunn; Salins; Zhou. Digital Health: Tracking Physiomes and Activity Using Wearable Biosensors Reveals Useful Health-Related Information. *PLoS Biol* (2017). doi:[10.1371/journal.pbio.2001402](https://doi.org/10.1371/journal.pbio.2001402)
14. Dunn; Kidzinski; Runge; Witt. Wearable sensors enable personalized predictions of clinical laboratory measurements. *Nat Med* (2021). doi:[10.1038/s41591-021-01339-0](https://doi.org/10.1038/s41591-021-01339-0)
15. Schüssler-Fiorenza Rose; Contrepois; Moneghetti; Zhou. A longitudinal big data approach for precision health. *Nat Med* (2019). doi:[10.1038/s41591-019-0414-6](https://doi.org/10.1038/s41591-019-0414-6)
16. Acciaroli; Sparacino; Hakaste; Facchinetti. Diabetes and Prediabetes Classification Using Glycemic Variability Indices From Continuous Glucose Monitoring Data. *J Diabetes Sci Technol* (2017). doi:[10.1177/1932296817710478](https://doi.org/10.1177/1932296817710478)
17. Longato; Acciaroli; Facchinetti; Maran. Simple Linear Support Vector Machine Classifier Can Distinguish Impaired Glucose Tolerance Versus Type 2 Diabetes Using a Reduced Set of CGM-Based Glycemic Variability Indices. *J Diabetes Sci Technol* (2019). doi:[10.1177/1932296819838856](https://doi.org/10.1177/1932296819838856)
18. Zeevi; Korem; Zmora; Israeli. Personalized Nutrition by Prediction of Glycemic Responses. *Cell* (2015). doi:[10.1016/j.cell.2015.11.001](https://doi.org/10.1016/j.cell.2015.11.001)
19. Marling; Shubrook; Vernier; Wiley. Characterizing blood glucose variability using new metrics with continuous glucose monitoring data. *J Diabetes Sci Technol* (2011). doi:[10.1177/193229681100500408](https://doi.org/10.1177/193229681100500408)
20. Fengade; Swati; Chandak; Rattan. Development of Enhanced Machine Learning Models for Predicting Type 2 Diabetes Mellitus Using Heart Rate Variability: A Retrospective Study. *Cureus* (2025). doi:[10.7759/cureus.80933](https://doi.org/10.7759/cureus.80933)
21. Kim; Yun; Kim; Matsushita. Heart rate circadian phase and hyperarousal as wearable digital phenotyping of insomnia: An interpretable machine learning study. *Digit Health* (2026). doi:[10.1177/20552076261458929](https://doi.org/10.1177/20552076261458929)
22. Kim; Mun; Lee. Detection and Analysis of Circadian Biomarkers for Metabolic Syndrome Using Wearable Data: Cross-Sectional Study. *JMIR Med Inform* (2025). doi:[10.2196/69328](https://doi.org/10.2196/69328)
23. Aggarwal; Das; Mazumder; Kumar. Heart rate variability time domain features in automated prediction of diabetes in rat. *Phys Eng Sci Med* (2020). doi:[10.1007/s13246-020-00950-8](https://doi.org/10.1007/s13246-020-00950-8)
24. Zhou; Chia; Wagner; Ayhan. A foundation model for generalizable disease detection from retinal images. *Nature* (2023). doi:[10.1038/s41586-023-06555-x](https://doi.org/10.1038/s41586-023-06555-x)
25. Gulshan; Peng; Coram; Stumpe. Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs. *JAMA* (2016). doi:[10.1001/jama.2016.17216](https://doi.org/10.1001/jama.2016.17216)
26. Dai; Sheng; Chen; Wu. A deep learning system for predicting time to progression of diabetic retinopathy. *Nat Med* (2024). doi:[10.1038/s41591-023-02702-z](https://doi.org/10.1038/s41591-023-02702-z)
27. Nusinovici; Rim; Li; Yu. Application of a deep-learning marker for morbidity and mortality prediction derived from retinal photographs: a cohort development and validation study. *Lancet Healthy Longev* (2024). doi:[10.1016/S2666-7568(24)00089-8](https://doi.org/10.1016/S2666-7568%2824%2900089-8)
28. Ponce-Bobadilla; Schmitt; Maier; Mensing. Practical guide to SHAP analysis: Explaining supervised machine learning model predictions in drug development. *Clin Transl Sci* (2024). doi:[10.1111/cts.70056](https://doi.org/10.1111/cts.70056)
29. Guan; Gong; Zhao; Yin. Interpretable machine learning model for new-onset atrial fibrillation prediction in critically ill patients: a multi-center study. *Crit Care* (2024). doi:[10.1186/s13054-024-05138-0](https://doi.org/10.1186/s13054-024-05138-0)
30. Li; Mansmann; Du; Hornung. Benchmark study of feature selection strategies for multi-omics data. *BMC Bioinformatics* (2022). doi:[10.1186/s12859-022-04962-x](https://doi.org/10.1186/s12859-022-04962-x)
31. Mi; Zou; Zou; Hu. Permutation-based identification of important biomarkers for complex diseases via machine learning models. *Nat Commun* (2021). doi:[10.1038/s41467-021-22756-2](https://doi.org/10.1038/s41467-021-22756-2)
32. Degenhardt; Seifert; Szymczak. Evaluation of variable selection methods for random forests and omics data sets. *Brief Bioinform* (2019). doi:[10.1093/bib/bbx124](https://doi.org/10.1093/bib/bbx124)
33. Tran; Castellano; Gutenkunst. Interpreting Supervised Machine Learning Inferences in Population Genomics Using Haplotype Matrix Permutations. *Mol Biol Evol* (2025). doi:[10.1093/molbev/msaf250](https://doi.org/10.1093/molbev/msaf250)
34. Cai; Lei; Roeder. Model-free prediction test with application to genomics data. *Proc Natl Acad Sci U S A* (2022). doi:[10.1073/pnas.2205518119](https://doi.org/10.1073/pnas.2205518119)
35. Chen; Guestrin. XGBoost: A Scalable Tree Boosting System. *KDD* (2016). doi:[10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
36. Lundberg; Lee. A Unified Approach to Interpreting Model Predictions. *NeurIPS (arXiv:1705.07874)* (2017). doi:[10.48550/arXiv.1705.07874](https://doi.org/10.48550/arXiv.1705.07874)
37. Meinshausen; Bühlmann. Stability selection. *J R Stat Soc B* (2010). doi:[10.1111/j.1467-9868.2010.00740.x](https://doi.org/10.1111/j.1467-9868.2010.00740.x)
38. Kursa; Rudnicki. Feature Selection with the Boruta Package. *J Stat Softw* (2010). doi:[10.18637/jss.v036.i11](https://doi.org/10.18637/jss.v036.i11)
39. Yang; Shi; Wei. MedMNIST v2, a large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Sci Data* (2023). doi:[10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)
40. He; Zhang; Ren; Sun. Deep Residual Learning for Image Recognition. *CVPR* (2016). doi:[10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
41. Oquab; Darcet; Moutakanni. DINOv2: Learning Robust Visual Features without Supervision. *TMLR (arXiv:2304.07193)* (2024). doi:[10.48550/arXiv.2304.07193](https://doi.org/10.48550/arXiv.2304.07193)
42. McMahan; Moore; Ramage. Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS (arXiv:1602.05629)* (2017). doi:[10.48550/arXiv.1602.05629](https://doi.org/10.48550/arXiv.1602.05629)
43. Dwork; Roth. The Algorithmic Foundations of Differential Privacy. *Found Trends Theor Comput Sci* (2014). doi:[10.1561/0400000042](https://doi.org/10.1561/0400000042)
44. Balle; Wang. Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising. *ICML (arXiv:1805.06530)* (2018). doi:[10.48550/arXiv.1805.06530](https://doi.org/10.48550/arXiv.1805.06530)

---

## BibTeX

```bibtex
@article{Huang2020,
  title   = {Fusion of medical imaging and electronic health records using deep learning: a systematic review and implementation guidelines.},
  author  = {Huang and Pareek and Seyyedi and Banerjee},
  journal = {NPJ Digit Med},
  year    = {2020},
  doi     = {10.1038/s41746-020-00341-z}
}

@article{Stahlschmidt2022,
  title   = {Multimodal deep learning for biomedical data fusion: a review.},
  author  = {Stahlschmidt and Ulfenborg and Synnergren},
  journal = {Brief Bioinform},
  year    = {2022},
  doi     = {10.1093/bib/bbab569}
}

@article{Lipkova2022,
  title   = {Artificial intelligence for multimodal data integration in oncology.},
  author  = {Lipkova and Chen and Chen and Lu},
  journal = {Cancer Cell},
  year    = {2022},
  doi     = {10.1016/j.ccell.2022.09.012}
}

@article{Ding2024,
  title   = {Large language multimodal models for new-onset type 2 diabetes prediction using five-year cohort electronic health records.},
  author  = {Ding and Thao and Peng and Wang},
  journal = {Sci Rep},
  year    = {2024},
  doi     = {10.1038/s41598-024-71020-2}
}

@article{Haleem2025,
  title   = {A multimodal deep learning architecture for predicting interstitial glucose for effective type 2 diabetes management.},
  author  = {Haleem and Katsarou and Georga and Dafoulas},
  journal = {Sci Rep},
  year    = {2025},
  doi     = {10.1038/s41598-025-07272-3}
}

@article{Argelaguet2018,
  title   = {Multi-Omics Factor Analysis-a framework for unsupervised integration of multi-omics data sets.},
  author  = {Argelaguet and Velten and Arnol and Dietrich},
  journal = {Mol Syst Biol},
  year    = {2018},
  doi     = {10.15252/msb.20178124}
}

@article{Singh2019,
  title   = {DIABLO: an integrative approach for identifying key molecular drivers from multi-omics assays.},
  author  = {Singh and Shannon and Gautier and Rohart},
  journal = {Bioinformatics},
  year    = {2019},
  doi     = {10.1093/bioinformatics/bty1054}
}

@article{Wang2014,
  title   = {Similarity network fusion for aggregating data types on a genomic scale.},
  author  = {Wang and Mezlini and Demir and Fiume},
  journal = {Nat Methods},
  year    = {2014},
  doi     = {10.1038/nmeth.2810}
}

@article{Allese2023,
  title   = {Discovery of drug-omics associations in type 2 diabetes with generative deep-learning models.},
  author  = {Allesøe and Lundgaard and Hernández Medina and Aguayo-Orozco},
  journal = {Nat Biotechnol},
  year    = {2023},
  doi     = {10.1038/s41587-022-01520-x}
}

@article{Yao2024,
  title   = {MOCAT: multi-omics integration with auxiliary classifiers enhanced autoencoder.},
  author  = {Yao and Jiang and Luo and Liang},
  journal = {BioData Min},
  year    = {2024},
  doi     = {10.1186/s13040-024-00360-6}
}

@article{Zhou2019,
  title   = {Longitudinal multi-omics of host-microbe dynamics in prediabetes.},
  author  = {Zhou and Sailani and Contrepois and Zhou},
  journal = {Nature},
  year    = {2019},
  doi     = {10.1038/s41586-019-1236-x}
}

@article{Hall2018,
  title   = {Glucotypes reveal new patterns of glucose dysregulation.},
  author  = {Hall and Perelman and Breschi and Limcaoco},
  journal = {PLoS Biol},
  year    = {2018},
  doi     = {10.1371/journal.pbio.2005143}
}

@article{Li2017,
  title   = {Digital Health: Tracking Physiomes and Activity Using Wearable Biosensors Reveals Useful Health-Related Information.},
  author  = {Li and Dunn and Salins and Zhou},
  journal = {PLoS Biol},
  year    = {2017},
  doi     = {10.1371/journal.pbio.2001402}
}

@article{Dunn2021,
  title   = {Wearable sensors enable personalized predictions of clinical laboratory measurements.},
  author  = {Dunn and Kidzinski and Runge and Witt},
  journal = {Nat Med},
  year    = {2021},
  doi     = {10.1038/s41591-021-01339-0}
}

@article{SchsslerFiorenzaRose2019,
  title   = {A longitudinal big data approach for precision health.},
  author  = {Schüssler-Fiorenza Rose and Contrepois and Moneghetti and Zhou},
  journal = {Nat Med},
  year    = {2019},
  doi     = {10.1038/s41591-019-0414-6}
}

@article{Acciaroli2017,
  title   = {Diabetes and Prediabetes Classification Using Glycemic Variability Indices From Continuous Glucose Monitoring Data.},
  author  = {Acciaroli and Sparacino and Hakaste and Facchinetti},
  journal = {J Diabetes Sci Technol},
  year    = {2017},
  doi     = {10.1177/1932296817710478}
}

@article{Longato2019,
  title   = {Simple Linear Support Vector Machine Classifier Can Distinguish Impaired Glucose Tolerance Versus Type 2 Diabetes Using a Reduced Set of CGM-Based Glycemic Variability Indices.},
  author  = {Longato and Acciaroli and Facchinetti and Maran},
  journal = {J Diabetes Sci Technol},
  year    = {2019},
  doi     = {10.1177/1932296819838856}
}

@article{Zeevi2015,
  title   = {Personalized Nutrition by Prediction of Glycemic Responses.},
  author  = {Zeevi and Korem and Zmora and Israeli},
  journal = {Cell},
  year    = {2015},
  doi     = {10.1016/j.cell.2015.11.001}
}

@article{Marling2011,
  title   = {Characterizing blood glucose variability using new metrics with continuous glucose monitoring data.},
  author  = {Marling and Shubrook and Vernier and Wiley},
  journal = {J Diabetes Sci Technol},
  year    = {2011},
  doi     = {10.1177/193229681100500408}
}

@article{Fengade2025,
  title   = {Development of Enhanced Machine Learning Models for Predicting Type 2 Diabetes Mellitus Using Heart Rate Variability: A Retrospective Study.},
  author  = {Fengade and Swati and Chandak and Rattan},
  journal = {Cureus},
  year    = {2025},
  doi     = {10.7759/cureus.80933}
}

@article{Kim2026,
  title   = {Heart rate circadian phase and hyperarousal as wearable digital phenotyping of insomnia: An interpretable machine learning study.},
  author  = {Kim and Yun and Kim and Matsushita},
  journal = {Digit Health},
  year    = {2026},
  doi     = {10.1177/20552076261458929}
}

@article{Kim2025,
  title   = {Detection and Analysis of Circadian Biomarkers for Metabolic Syndrome Using Wearable Data: Cross-Sectional Study.},
  author  = {Kim and Mun and Lee},
  journal = {JMIR Med Inform},
  year    = {2025},
  doi     = {10.2196/69328}
}

@article{Aggarwal2020,
  title   = {Heart rate variability time domain features in automated prediction of diabetes in rat.},
  author  = {Aggarwal and Das and Mazumder and Kumar},
  journal = {Phys Eng Sci Med},
  year    = {2020},
  doi     = {10.1007/s13246-020-00950-8}
}

@article{Zhou2023,
  title   = {A foundation model for generalizable disease detection from retinal images.},
  author  = {Zhou and Chia and Wagner and Ayhan},
  journal = {Nature},
  year    = {2023},
  doi     = {10.1038/s41586-023-06555-x}
}

@article{Gulshan2016,
  title   = {Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs.},
  author  = {Gulshan and Peng and Coram and Stumpe},
  journal = {JAMA},
  year    = {2016},
  doi     = {10.1001/jama.2016.17216}
}

@article{Dai2024,
  title   = {A deep learning system for predicting time to progression of diabetic retinopathy.},
  author  = {Dai and Sheng and Chen and Wu},
  journal = {Nat Med},
  year    = {2024},
  doi     = {10.1038/s41591-023-02702-z}
}

@article{Nusinovici2024,
  title   = {Application of a deep-learning marker for morbidity and mortality prediction derived from retinal photographs: a cohort development and validation study.},
  author  = {Nusinovici and Rim and Li and Yu},
  journal = {Lancet Healthy Longev},
  year    = {2024},
  doi     = {10.1016/S2666-7568(24)00089-8}
}

@article{PonceBobadilla2024,
  title   = {Practical guide to SHAP analysis: Explaining supervised machine learning model predictions in drug development.},
  author  = {Ponce-Bobadilla and Schmitt and Maier and Mensing},
  journal = {Clin Transl Sci},
  year    = {2024},
  doi     = {10.1111/cts.70056}
}

@article{Guan2024,
  title   = {Interpretable machine learning model for new-onset atrial fibrillation prediction in critically ill patients: a multi-center study.},
  author  = {Guan and Gong and Zhao and Yin},
  journal = {Crit Care},
  year    = {2024},
  doi     = {10.1186/s13054-024-05138-0}
}

@article{Li2022,
  title   = {Benchmark study of feature selection strategies for multi-omics data.},
  author  = {Li and Mansmann and Du and Hornung},
  journal = {BMC Bioinformatics},
  year    = {2022},
  doi     = {10.1186/s12859-022-04962-x}
}

@article{Mi2021,
  title   = {Permutation-based identification of important biomarkers for complex diseases via machine learning models.},
  author  = {Mi and Zou and Zou and Hu},
  journal = {Nat Commun},
  year    = {2021},
  doi     = {10.1038/s41467-021-22756-2}
}

@article{Degenhardt2019,
  title   = {Evaluation of variable selection methods for random forests and omics data sets.},
  author  = {Degenhardt and Seifert and Szymczak},
  journal = {Brief Bioinform},
  year    = {2019},
  doi     = {10.1093/bib/bbx124}
}

@article{Tran2025,
  title   = {Interpreting Supervised Machine Learning Inferences in Population Genomics Using Haplotype Matrix Permutations.},
  author  = {Tran and Castellano and Gutenkunst},
  journal = {Mol Biol Evol},
  year    = {2025},
  doi     = {10.1093/molbev/msaf250}
}

@article{Cai2022,
  title   = {Model-free prediction test with application to genomics data.},
  author  = {Cai and Lei and Roeder},
  journal = {Proc Natl Acad Sci U S A},
  year    = {2022},
  doi     = {10.1073/pnas.2205518119}
}

@inproceedings{Chen2016,
  title   = {XGBoost: A Scalable Tree Boosting System},
  author  = {Chen and Guestrin},
  booktitle = {KDD},
  year    = {2016},
  doi     = {10.1145/2939672.2939785}
}

@inproceedings{Lundberg2017,
  title   = {A Unified Approach to Interpreting Model Predictions},
  author  = {Lundberg and Lee},
  booktitle = {NeurIPS (arXiv:1705.07874)},
  year    = {2017},
  doi     = {10.48550/arXiv.1705.07874}
}

@article{Meinshausen2010,
  title   = {Stability selection},
  author  = {Meinshausen and Bühlmann},
  journal = {J R Stat Soc B},
  year    = {2010},
  doi     = {10.1111/j.1467-9868.2010.00740.x}
}

@article{Kursa2010,
  title   = {Feature Selection with the Boruta Package},
  author  = {Kursa and Rudnicki},
  journal = {J Stat Softw},
  year    = {2010},
  doi     = {10.18637/jss.v036.i11}
}

@article{Yang2023,
  title   = {MedMNIST v2 — a large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author  = {Yang and Shi and Wei},
  journal = {Sci Data},
  year    = {2023},
  doi     = {10.1038/s41597-022-01721-8}
}

@inproceedings{He2016,
  title   = {Deep Residual Learning for Image Recognition},
  author  = {He and Zhang and Ren and Sun},
  booktitle = {CVPR},
  year    = {2016},
  doi     = {10.1109/CVPR.2016.90}
}

@inproceedings{Oquab2024,
  title   = {DINOv2: Learning Robust Visual Features without Supervision},
  author  = {Oquab and Darcet and Moutakanni},
  booktitle = {TMLR (arXiv:2304.07193)},
  year    = {2024},
  doi     = {10.48550/arXiv.2304.07193}
}

@inproceedings{McMahan2017,
  title   = {Communication-Efficient Learning of Deep Networks from Decentralized Data},
  author  = {McMahan and Moore and Ramage},
  booktitle = {AISTATS (arXiv:1602.05629)},
  year    = {2017},
  doi     = {10.48550/arXiv.1602.05629}
}

@article{Dwork2014,
  title   = {The Algorithmic Foundations of Differential Privacy},
  author  = {Dwork and Roth},
  journal = {Found Trends Theor Comput Sci},
  year    = {2014},
  doi     = {10.1561/0400000042}
}

@inproceedings{Balle2018,
  title   = {Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising},
  author  = {Balle and Wang},
  booktitle = {ICML (arXiv:1805.06530)},
  year    = {2018},
  doi     = {10.48550/arXiv.1805.06530}
}
```
