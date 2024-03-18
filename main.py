from config import load_config
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='conf.yml', help='path to the config.yaml file')
    args = parser.parse_args()
    config = load_config(args.config)
    print('Config loaded')
    mode = config.MODE
    task = config.TASK
    if task == "Deblurring":
        from Deblurring.src.trainer import Trainer
        from Deblurring.src.tester import Tester
        from Deblurring.src.finetune import Finetune
    else:
        from Binarization.src.trainer import Trainer
        from Binarization.src.tester import Tester

    if mode == 0:
        print("--------------------------")
        print('Start Testing')
        print("--------------------------")

        tester = Tester(config)
        tester.test()

        print("--------------------------")
        print('Testing complete')
        print("--------------------------")

    elif mode == 1:


        print("--------------------------")
        print('Start Training')
        print("--------------------------")

        trainer = Trainer(config)
        trainer.train()

        print("--------------------------")
        print('Training complete')
        print("--------------------------")


    else: 
        print("--------------------------")
        print('Start Finetuning')
        print("--------------------------")

        finetuner = Finetune(config)
        finetuner.finetune()

        print("--------------------------")
        print('Finetuning complete')
        print("--------------------------")


        
if __name__ == "__main__":
    main()