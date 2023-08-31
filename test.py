from engines.test_engine import TestEngine
import argparse
from models.factory import create_test_model


def main(args):
    model = create_test_model(args.model)
    test_engine = TestEngine(model=model, testset_file_path=args.testset,
                             test_type=args.test_type, dist_fn=args.dist_fn,
                             threshold=args.threshold)
    
    recall, precision, f1_score = test_engine.test()
    print(f'model: {args.model}, checkpoint: {args.checkpoint}')
    print(f'test_type: {args.test_type}, dist_fn: {args.dist_fn}, threshold: {args.threshold}')
    print(f'Recall: {recall:.4f}, Precision: {precision:.4f}, F1 score: {f1_score:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--ckpt', type=str, help='Model checkpoint path')
    parser.add_argument('--testset', type=str, help='Testset csv file path')
    parser.add_argument('--test_type', type=str, default='id_only', help='Choose from "all", "gesture_only" and "id_only"')
    parser.add_argument('--dist_fn', type=str, default='euclidean', help='Choose from "cosine" and "euclidean"')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for cosine distance')
    args = parser.parse_args()
    main(args)