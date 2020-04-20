#!/usr/bin/env bash
export PYTHONPATH=.

# k: number of neighbors, w: ['uniform', 'distance'], p: [None, 'normalize', 'scale', 'min_max_scale'] - pre process method
# fs: [None, 'k_best', 'decision_trees', 'drop_correlated'], fp: characteristic for feature selection (K>1for k_best, 0<threshold<1 for drop_correlated)
# s: [None, 'grid_search']

# BREAST CANCER - KNN
# run without feature selection on raw data
qsub -cwd -N KNNBreastCancer_grid_search_no_feature_selection -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="None" fs="None" fp="None" s="grid_search";

# run with feature selection
qsub -cwd -N KNNBreastCancer_grid_search_feature_selection_k_best_10 -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="None" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N KNNBreastCancer_grid_search_feature_selection_decision_trees -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="None" fs="decision_trees" fp="15" s="grid_search";

qsub -cwd -N KNNBreastCancer_grid_search_feature_selection_drop_correlated_08 -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="None" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing and no feature selection
qsub -cwd -N KNNBreastCancer_grid_search_nofeature_normalized -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="normalize" fs="None" fp="None" s="grid_search";

qsub -cwd -N KNNBreastCancer_grid_search_nofeature_scaled -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="scale" fs="None" fp="None" s="grid_search";

qsub -cwd -N KNNBreastCancer_grid_search_nofeature_minmaxscaled -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="min_max_scale" fs="None" fp="None" s="grid_search";


# run with data preprocessing (normalization) and feature selection
qsub -cwd -N KNNBreastCancer_grid_search_feature_selection_k_best_10_normalized -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="normalize" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N KNNBreastCancer_grid_search_feature_selection_decision_trees_normalized -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="normalize" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N KNNBreastCancer_grid_search_feature_selection_drop_correlated_08_normalized -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="normalize" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (scaling) and feature selection
qsub -cwd -N KNNBreastCancer_grid_search_feature_selection_k_best_10_scaled -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="scale" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N KNNBreastCancer_grid_search_feature_selection_decision_trees_scaled -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N KNNBreastCancer_grid_search_feature_selection_drop_correlated_08_scaled -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="scale" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (min_max_scaling) and feature selection
qsub -cwd -N KNNBreastCancer_grid_search_feature_selection_k_best_10_minmaxscale -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="min_max_scale" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N KNNBreastCancer_grid_search_feature_selection_decision_trees_minmaxscale -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="min_max_scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N KNNBreastCancer_grid_search_feature_selection_drop_correlated_08_minmaxscale -e job_reports/ -o job_reports/ run-knn-breast-cancer.sh k=3 w="uniform" p="min_max_scale" fs="drop_correlated" fp="0.8" s="grid_search";


# BANK MARKETING - KNN
# run without feature selection on raw data
qsub -cwd -N KNNBankMarketing_grid_search_no_feature_selection -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="None" fs="None" fp="None" s="grid_search";

# run with feature selection
qsub -cwd -N KNNBankMarketing_grid_search_feature_selection_k_best_10 -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="None" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N KNNBankMarketing_grid_search_feature_selection_decision_trees -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="None" fs="decision_trees" fp="15" s="grid_search";

qsub -cwd -N KNNBankMarketing_grid_search_feature_selection_drop_correlated_08 -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="None" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing and no feature selection
qsub -cwd -N KNNBankMarketing_grid_search_nofeature_normalized -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="normalize" fs="None" fp="None" s="grid_search";

qsub -cwd -N KNNBankMarketing_grid_search_nofeature_scaled -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="scale" fs="None" fp="None" s="grid_search";

qsub -cwd -N KNNBankMarketing_grid_search_nofeature_minmaxscaled -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="min_max_scale" fs="None" fp="None" s="grid_search";


# run with data preprocessing (normalization) and feature selection
qsub -cwd -N KNNBankMarketing_grid_search_feature_selection_k_best_10_normalized -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="normalize" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N KNNBankMarketing_grid_search_feature_selection_decision_trees_normalized -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="normalize" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N KNNBankMarketing_grid_search_feature_selection_drop_correlated_08_normalized -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="normalize" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (scaling) and feature selection
qsub -cwd -N KNNBankMarketing_grid_search_feature_selection_k_best_10_scaled -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="scale" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N KNNBankMarketing_grid_search_feature_selection_decision_trees_scaled -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N KNNBankMarketing_grid_search_feature_selection_drop_correlated_08_scaled -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="scale" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (min_max_scaling) and feature selection
qsub -cwd -N KNNBankMarketing_grid_search_feature_selection_k_best_10_minmaxscale -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="min_max_scale" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N KNNBankMarketing_grid_search_feature_selection_decision_trees_minmaxscale -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="min_max_scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N KNNBankMarketingKNNBankMarketing_grid_search_feature_selection_drop_correlated_08_minmaxscale -e job_reports/ -o job_reports/ run-knn-bank-marketing.sh k=3 w="uniform" p="min_max_scale" fs="drop_correlated" fp="0.8" s="grid_search";


# IMAGE-SEGMENTATION - KNN
# run without feature selection on raw data
qsub -cwd -N KNNImageSegmentation_grid_search_no_feature_selection -e job_reports/ -o job_reports/ run-knn-image-segmentation.sh k=3 w="uniform" p="None" fs="None" fp="None" s="grid_search";

# run with feature selection
qsub -cwd -N KNNImageSegmentation_grid_search_feature_selection_decision_trees -e job_reports/ -o job_reports/ run-knn-image-segmentation.sh k=3 w="uniform" p="None" fs="decision_trees" fp="15" s="grid_search";

qsub -cwd -N KNNImageSegmentation_grid_search_feature_selection_drop_correlated_08 -e job_reports/ -o job_reports/ run-knn-image-segmentation.sh k=3 w="uniform" p="None" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing and no feature selection
qsub -cwd -N KNNImageSegmentation_grid_search_nofeature_normalized -e job_reports/ -o job_reports/ run-knn-image-segmentation.sh k=3 w="uniform" p="normalize" fs="None" fp="None" s="grid_search";

qsub -cwd -N KNNImageSegmentation_grid_search_nofeature_scaled -e job_reports/ -o job_reports/ run-knn-image-segmentation.sh k=3 w="uniform" p="scale" fs="None" fp="None" s="grid_search";

qsub -cwd -N KNNImageSegmentation_grid_search_nofeature_minmaxscaled -e job_reports/ -o job_reports/ run-knn-image-segmentation.sh k=3 w="uniform" p="min_max_scale" fs="None" fp="None" s="grid_search";


# run with data preprocessing (normalization) and feature selection
qsub -cwd -N KNNImageSegmentation_grid_search_feature_selection_decision_trees_normalized -e job_reports/ -o job_reports/ run-knn-image-segmentation.sh k=3 w="uniform" p="normalize" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N KNNImageSegmentation_grid_search_feature_selection_drop_correlated_08_normalized -e job_reports/ -o job_reports/ run-knn-image-segmentation.sh k=3 w="uniform" p="normalize" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (scaling) and feature selection
qsub -cwd -N KNNImageSegmentation_grid_search_feature_selection_decision_trees_scaled -e job_reports/ -o job_reports/ run-knn-image-segmentation.sh k=3 w="uniform" p="scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N KNNImageSegmentation_grid_search_feature_selection_drop_correlated_08_scaled -e job_reports/ -o job_reports/ run-knn-image-segmentation.sh k=3 w="uniform" p="scale" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (min_max_scaling) and feature selection
qsub -cwd -N KNNImageSegmentation_grid_search_feature_selection_decision_trees_minmaxscale -e job_reports/ -o job_reports/ run-knn-image-segmentation.sh k=3 w="uniform" p="min_max_scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N KNNImageSegmentation_grid_search_feature_selection_drop_correlated_08_minmaxscale -e job_reports/ -o job_reports/ run-knn-image-segmentation.sh k=3 w="uniform" p="min_max_scale" fs="drop_correlated" fp="0.8" s="grid_search";


# BREAST CANCER - Decision Trees
# run without feature selection on raw data
qsub -cwd -N DTBreastCancer_grid_search_no_feature_selection -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="None" fs="None" fp="None" s="grid_search";

# run with feature selection
qsub -cwd -N DTBreastCancer_grid_search_feature_selection_k_best_10 -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="None" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N DTBreastCancer_grid_search_feature_selection_decision_trees -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="None" fs="decision_trees" fp="15" s="grid_search";

qsub -cwd -N DTBreastCancerr_grid_search_feature_selection_drop_correlated_08 -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="None" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing and no feature selection
qsub -cwd -N DTBreastCancer_grid_search_nofeature_normalized -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="normalize" fs="None" fp="None" s="grid_search";

qsub -cwd -N DTBreastCancer_grid_search_nofeature_scaled -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="scale" fs="None" fp="None" s="grid_search";

qsub -cwd -N DTBreastCancer_grid_search_nofeature_minmaxscaled -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="min_max_scale" fs="None" fp="None" s="grid_search";


# run with data preprocessing (normalization) and feature selection
qsub -cwd -N DTBreastCancer_grid_search_feature_selection_k_best_10_normalized -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="normalize" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N DTBreastCancer_grid_search_feature_selection_decision_trees_normalized -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="normalize" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N DTBreastCancer_grid_search_feature_selection_drop_correlated_08_normalized -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="normalize" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (scaling) and feature selection
qsub -cwd -N DTBreastCancer_grid_search_feature_selection_k_best_10_scaled -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="scale" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N DTBreastCancer_grid_search_feature_selection_decision_trees_scaled -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N DTBreastCancer_grid_search_feature_selection_drop_correlated_08_scaled -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="scale" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (min_max_scaling) and feature selection
qsub -cwd -N DTBreastCancer_grid_search_feature_selection_k_best_10_minmaxscale -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="min_max_scale" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N DTBreastCancer_grid_search_feature_selection_decision_trees_minmaxscale -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="min_max_scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N DTBreastCancer_grid_search_feature_selection_drop_correlated_08_minmaxscale -e job_reports/ -o job_reports/ run-decision-trees-breast-cancer.sh p="min_max_scale" fs="drop_correlated" fp="0.8" s="grid_search";


# BANK MARKETING - Decision Trees
# run without feature selection on raw data
qsub -cwd -N DTBankMarketing_grid_search_no_feature_selection -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="None" fs="None" fp="None" s="grid_search";

# run with feature selection
qsub -cwd -N DTBankMarketing_grid_search_feature_selection_k_best_10 -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="None" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N DTBankMarketing_grid_search_feature_selection_decision_trees -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="None" fs="decision_trees" fp="15" s="grid_search";

qsub -cwd -N DTBankMarketingr_grid_search_feature_selection_drop_correlated_08 -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="None" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing and no feature selection
qsub -cwd -N DTBankMarketing_grid_search_nofeature_normalized -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="normalize" fs="None" fp="None" s="grid_search";

qsub -cwd -N DTBankMarketing_grid_search_nofeature_scaled -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="scale" fs="None" fp="None" s="grid_search";

qsub -cwd -N DTBankMarketing_grid_search_nofeature_minmaxscaled -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="min_max_scale" fs="None" fp="None" s="grid_search";


# run with data preprocessing (normalization) and feature selection
qsub -cwd -N DTBankMarketing_grid_search_feature_selection_k_best_10_normalized -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="normalize" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N DTBankMarketing_grid_search_feature_selection_decision_trees_normalized -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="normalize" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N DTBankMarketing_grid_search_feature_selection_drop_correlated_08_normalized -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="normalize" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (scaling) and feature selection
qsub -cwd -N DTBankMarketing_grid_search_feature_selection_k_best_10_scaled -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="scale" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N DTBankMarketing_grid_search_feature_selection_decision_trees_scaled -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N DTBankMarketing_grid_search_feature_selection_drop_correlated_08_scaled -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="scale" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (min_max_scaling) and feature selection
qsub -cwd -N DTBankMarketing_grid_search_feature_selection_k_best_10_minmaxscale -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="min_max_scale" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N DTBankMarketing_grid_search_feature_selection_decision_trees_minmaxscale -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="min_max_scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N DTBankMarketing_grid_search_feature_selection_drop_correlated_08_minmaxscale -e job_reports/ -o job_reports/ run-decision-trees-bank_marketing.sh p="min_max_scale" fs="drop_correlated" fp="0.8" s="grid_search";


# IMAGE SEGMENTATION - Decision Trees
# run without feature selection on raw data
qsub -cwd -N DTImageSegmentation_grid_search_no_feature_selection -e job_reports/ -o job_reports/ run-decision-trees-image-segmentation.sh p="None" fs="None" fp="None" s="grid_search";

# run with feature selection
qsub -cwd -N DTImageSegmentation_grid_search_feature_selection_decision_trees -e job_reports/ -o job_reports/ run-decision-trees-image-segmentation.sh p="None" fs="decision_trees" fp="15" s="grid_search";

qsub -cwd -N DTImageSegmentationr_grid_search_feature_selection_drop_correlated_08 -e job_reports/ -o job_reports/ run-decision-trees-image-segmentation.sh p="None" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing and no feature selection
qsub -cwd -N DTImageSegmentation_grid_search_nofeature_normalized -e job_reports/ -o job_reports/ run-decision-trees-image-segmentation.sh p="normalize" fs="None" fp="None" s="grid_search";

qsub -cwd -N DTImageSegmentation_grid_search_nofeature_scaled -e job_reports/ -o job_reports/ run-decision-trees-image-segmentation.sh p="scale" fs="None" fp="None" s="grid_search";

qsub -cwd -N DTImageSegmentation_grid_search_nofeature_minmaxscaled -e job_reports/ -o job_reports/ run-decision-trees-image-segmentation.sh p="min_max_scale" fs="None" fp="None" s="grid_search";


# run with data preprocessing (normalization) and feature selection
qsub -cwd -N DTImageSegmentation_grid_search_feature_selection_decision_trees_normalized -e job_reports/ -o job_reports/ run-decision-trees-image-segmentation.sh p="normalize" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N DTImageSegmentation_grid_search_feature_selection_drop_correlated_08_normalized -e job_reports/ -o job_reports/ run-decision-trees-image-segmentation.sh p="normalize" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (scaling) and feature selection
qsub -cwd -N DTImageSegmentation_grid_search_feature_selection_decision_trees_scaled -e job_reports/ -o job_reports/ run-decision-trees-image-segmentation.sh p="scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N DTImageSegmentation_grid_search_feature_selection_drop_correlated_08_scaled -e job_reports/ -o job_reports/ run-decision-trees-image-segmentation.sh p="scale" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (min_max_scaling) and feature selection
qsub -cwd -N DTImageSegmentation_grid_search_feature_selection_decision_trees_minmaxscale -e job_reports/ -o job_reports/ run-decision-trees-image-segmentation.sh p="min_max_scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N DTImageSegmentation_grid_search_feature_selection_drop_correlated_08_minmaxscale -e job_reports/ -o job_reports/ run-decision-trees-image-segmentation.sh p="min_max_scale" fs="drop_correlated" fp="0.8" s="grid_search";


# BREAST_CANCER - NaiveBayes
# run without feature selection on raw data
qsub -cwd -N NBBreastCancer_grid_search_no_feature_selection -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="None" fs="None" fp="None" s="grid_search";

# run with feature selection
qsub -cwd -N NBBreastCancer_grid_search_feature_selection_k_best_10 -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="None" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N NBBreastCancer_grid_search_feature_selection_decision_trees -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="None" fs="decision_trees" fp="15" s="grid_search";

qsub -cwd -N NBBreastCancerr_grid_search_feature_selection_drop_correlated_08 -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="None" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing and no feature selection
qsub -cwd -N NBBreastCancer_grid_search_nofeature_normalized -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="normalize" fs="None" fp="None" s="grid_search";

qsub -cwd -N NBBreastCancer_grid_search_nofeature_scaled -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="scale" fs="None" fp="None" s="grid_search";

qsub -cwd -N NBBreastCancer_grid_search_nofeature_minmaxscaled -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="min_max_scale" fs="None" fp="None" s="grid_search";


# run with data preprocessing (normalization) and feature selection
qsub -cwd -N NBBreastCancer_grid_search_feature_selection_k_best_10_normalized -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="normalize" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N NBBreastCancer_grid_search_feature_selection_decision_trees_normalized -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="normalize" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N NBBreastCancer_grid_search_feature_selection_drop_correlated_08_normalized -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="normalize" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (scaling) and feature selection
qsub -cwd -N NBBreastCancer_grid_search_feature_selection_k_best_10_scaled -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="scale" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N NBBreastCancer_grid_search_feature_selection_decision_trees_scaled -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N NBBreastCancer_grid_search_feature_selection_drop_correlated_08_scaled -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="scale" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (min_max_scaling) and feature selection
qsub -cwd -N NBBreastCancer_grid_search_feature_selection_k_best_10_minmaxscale -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="min_max_scale" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N NBBreastCancer_grid_search_feature_selection_decision_trees_minmaxscale -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="min_max_scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N NBBreastCancer_grid_search_feature_selection_drop_correlated_08_minmaxscale -e job_reports/ -o job_reports/ run-naive-bayes-breast-cancer.sh p="min_max_scale" fs="drop_correlated" fp="0.8" s="grid_search";



# BREAST_CANCER - RandomForest
# run without feature selection on raw data
qsub -cwd -N RFBreastCancer_grid_search_no_feature_selection -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="None" fs="None" fp="None" s="grid_search";

# run with feature selection
qsub -cwd -N RFBreastCancer_grid_search_feature_selection_k_best_10 -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="None" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N RFBreastCancer_grid_search_feature_selection_decision_trees -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="None" fs="decision_trees" fp="15" s="grid_search";

qsub -cwd -N RFBreastCancerr_grid_search_feature_selection_drop_correlated_08 -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="None" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing and no feature selection
qsub -cwd -N RFBreastCancer_grid_search_nofeature_normalized -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="normalize" fs="None" fp="None" s="grid_search";

qsub -cwd -N RFBreastCancer_grid_search_nofeature_scaled -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="scale" fs="None" fp="None" s="grid_search";

qsub -cwd -N RFBreastCancer_grid_search_nofeature_minmaxscaled -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="min_max_scale" fs="None" fp="None" s="grid_search";


# run with data preprocessing (normalization) and feature selection
qsub -cwd -N RFBreastCancer_grid_search_feature_selection_k_best_10_normalized -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="normalize" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N RFBreastCancer_grid_search_feature_selection_decision_trees_normalized -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="normalize" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N RFBreastCancer_grid_search_feature_selection_drop_correlated_08_normalized -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="normalize" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (scaling) and feature selection
qsub -cwd -N RFBreastCancer_grid_search_feature_selection_k_best_10_scaled -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="scale" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N RFBreastCancer_grid_search_feature_selection_decision_trees_scaled -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N RFBreastCancer_grid_search_feature_selection_drop_correlated_08_scaled -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="scale" fs="drop_correlated" fp="0.8" s="grid_search";

# run with data preprocessing (min_max_scaling) and feature selection
qsub -cwd -N RFBreastCancer_grid_search_feature_selection_k_best_10_minmaxscale -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="min_max_scale" fs="k_best" fp="10" s="grid_search";

qsub -cwd -N RFBreastCancer_grid_search_feature_selection_decision_trees_minmaxscale -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="min_max_scale" fs="decision_trees" fp="None" s="grid_search";

qsub -cwd -N RFBreastCancer_grid_search_feature_selection_drop_correlated_08_minmaxscale -e job_reports/ -o job_reports/ run-random-forest-breast-cancer.sh p="min_max_scale" fs="drop_correlated" fp="0.8" s="grid_search";