"""รันทั้ง pipeline: python src/main.py

1. preprocess.py  data/raw → data/processed
2. analysis.py    EDA + โมเดล + วิเคราะห์ความผิดพลาด → results/figures, results/output
"""
import sys

import analysis
import preprocess


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print('== Step 1: preprocess ==')
    preprocess.run()
    print('\n== Step 2: analysis & modeling ==')
    analysis.run()
    print('\nDone. See results/figures and results/output')


if __name__ == '__main__':
    main()
