#!/usr/bin/env bash
export PYTHONPATH=.

# declare variables for arguments
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in k)              k=${VALUE} ;;
            w)    w=${VALUE} ;;
            p)    p=${VALUE} ;;
            fs)    fs=${VALUE} ;;
            fp)    fp=${VALUE} ;;
            s)    s=${VALUE} ;;
            *)
    esac
done

python3.7 decision_trees_breast_cancer.py --preprocess_method $p --feature_selection $fs --feature_param $fp --search_params $s
