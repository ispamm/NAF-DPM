import os
from Binarization.schedule.schedule import Schedule
from Binarization.model.NAFDPM import NAFDPM, EMA
import utils.util as util
from utils.util import crop_concat, crop_concat_back, min_max
from Binarization.schedule.diffusionSample import GaussianDiffusion
from Binarization.schedule.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import copy
from Binarization.src.sobel import Laplacian
import logging
from collections import OrderedDict
import pyiqa
import wandb
from utils.metrics import compute_metrics_DIBCO


class Trainer:
    def __init__(self, config):
        torch.manual_seed(0)
        self.mode = config.MODE
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #DEFINE NETWORK
        in_channels = config.CHANNEL_X
        out_channels = config.CHANNEL_Y
        self.out_channels = out_channels
        self.network = NAFDPM(input_channels=in_channels,
            output_channels = out_channels,
            n_channels      = config.MODEL_CHANNELS,
            middle_blk_num  = config.MIDDLE_BLOCKS, 
            enc_blk_nums    = config.ENC_BLOCKS, 
            dec_blk_nums    = config.DEC_BLOCKS).to(self.device)

        self.bestPSNR = 0
        self.bestFmeasure = 0
        self.bestPFmeasure = 0
        
        #DIFFUSION SAMPLING USING GAUSSIAN DIFFUSION
        self.schedule = Schedule(config.SCHEDULE, config.TIMESTEPS)
        self.diffusion = GaussianDiffusion(self.network.denoiser, config.TIMESTEPS, self.schedule).to(self.device)

        #LOGGER AND PATHS
        self.test_img_save_path = config.TEST_IMG_SAVE_PATH
        self.logger_path = config.LOGGER_PATH
        if not os.path.exists(self.test_img_save_path):
            os.makedirs(self.test_img_save_path)
        if not os.path.exists(self.logger_path):
            os.makedirs(self.logger_path)
        util.setup_logger(
               "base",
                config.LOGGER_PATH,
                "train" + "DocDiff",
                level=logging.INFO,
                screen=True,
                tofile=True,
            )
        self.logger = logging.getLogger("base")

        self.training_path = config.TRAINING_PATH
        if not os.path.exists(self.training_path):
            os.makedirs(self.training_path)

        self.pretrained_path_init_predictor = config.PRETRAINED_PATH_INITIAL_PREDICTOR
        self.pretrained_path_denoiser = config.PRETRAINED_PATH_DENOISER
        self.continue_training = config.CONTINUE_TRAINING
        self.continue_training_steps = 0
        self.path_train_gt = config.PATH_GT
        self.path_train_img = config.PATH_IMG
        self.weight_save_path = config.WEIGHT_SAVE_PATH
        self.test_path_img = config.TEST_PATH_IMG
        self.test_path_gt = config.TEST_PATH_GT
        self.save_img_path = util.init__result_Dir(self.training_path)

        #LR ITERATIONS AND TRAINING STUFFS
        self.iteration_max = config.ITERATION_MAX
        self.LR = config.LR
        self.cross_entropy = nn.BCELoss()
        self.num_timesteps = config.TIMESTEPS
        self.ema_every = config.EMA_EVERY
        self.start_ema = config.START_EMA
        self.save_model_every = config.SAVE_MODEL_EVERY
        self.EMA_or_not = config.EMA
        self.TEST_INITIAL_PREDICTOR_WEIGHT_PATH = config.TEST_INITIAL_PREDICTOR_WEIGHT_PATH
        self.TEST_DENOISER_WEIGHT_PATH = config.TEST_DENOISER_WEIGHT_PATH
        self.DPM_SOLVER = config.DPM_SOLVER
        self.DPM_STEP = config.DPM_STEP
        self.beta_loss = config.BETA_LOSS
        self.pre_ori = config.PRE_ORI
        self.high_low_freq = config.HIGH_LOW_FREQ
        self.image_size = config.IMAGE_SIZE
        self.native_resolution = config.NATIVE_RESOLUTION
        self.validate_every = config.VALIDATE_EVERY
        self.optimizer = optim.AdamW(self.network.parameters(), lr=self.LR, weight_decay=1e-4)
        self.val_iterations = config.VALIDATE_ITERATIONS

 
        #DATASETS AND DATALOADERS
        from Binarization.data.docdata import DocData
        if self.mode == 1:
            dataset_train = DocData(self.path_train_img, self.path_train_gt, config.IMAGE_SIZE, self.mode)
            self.batch_size = config.BATCH_SIZE
            self.dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                               num_workers=config.NUM_WORKERS)
            dataset_test = DocData(config.TEST_PATH_IMG, config.TEST_PATH_GT, config.IMAGE_SIZE, 0)
            self.dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE_VAL, shuffle=False,
                                              drop_last=False,
                                              num_workers=config.NUM_WORKERS)
        else:
            dataset_test = DocData(config.TEST_PATH_IMG, config.TEST_PATH_GT, config.IMAGE_SIZE, self.mode)
            self.dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE_VAL, shuffle=False,
                                              drop_last=False,
                                              num_workers=config.NUM_WORKERS)
        if self.mode == 1 and self.continue_training == 'True':
            print('Continue Training')
            checkpoint_init = torch.load(self.pretrained_path_init_predictor)
            checkpoint_denoiser = torch.load(self.pretrained_path_denoiser)
            self.network.init_predictor.load_state_dict(checkpoint_init['model_state_dict'])
            self.network.denoiser.load_state_dict(checkpoint_denoiser['model_state_dict'])
            self.continue_training_steps = checkpoint_denoiser['iteration']
            self.optimizer.load_state_dict(checkpoint_denoiser['optimizer_state_dict'])
            self.bestPSNR = checkpoint_denoiser['bestPSNR']
            self.bestFmeasure = checkpoint_denoiser['bestFmeasure']
            self.bestPFmeasure = checkpoint_denoiser['bestPFmeasure']

            
        if self.mode == 1 and config.EMA == 'True':
            self.EMA = EMA(0.9999)
            self.ema_model = copy.deepcopy(self.network).to(self.device)
        if config.LOSS == 'L1':
            self.loss = nn.L1Loss()
        elif config.LOSS == 'L2':
            self.loss = nn.MSELoss()
        else:
            print('Loss not implemented, setting the loss to L2 (default one)')
            self.loss = nn.MSELoss()
        if self.high_low_freq == 'True':
            self.high_filter = Laplacian().to(self.device)
        
        #WANDB LOGIN AND SET UP
        self.wandb = config.WANDB
        if self.wandb == "True":
            self.wandb = True
            wandb.login()
            run = wandb.init(
                # Set the project where this run will be logged
                project=config.PROJECT,
                # Track hyperparameters and run metadata
                config={
                    "learning_rate": self.LR,
                    "iterations": self.iteration_max,
                    "Native": self.native_resolution,
                    "DPM_Solver": self.DPM_SOLVER,
                    "Sampling_Steps": config.TIMESTEPS
            })
        else:
            self.wandb = False 

        #DEFINE METRICS
        self.ssim = pyiqa.create_metric('ssim', device=self.device)
        if self.wandb:
            wandb.define_metric("psnr", summary="max")
            wandb.define_metric("ssim", summary="max")
            wandb.define_metric("fmeasure", summary="max")
            wandb.define_metric("pfmeasure", summary="max")
            wandb.define_metric("drd", summary="max")


    # VALIDATE FUNCTION
    def validate(self,current_iteration):

        with torch.no_grad():
            #PUT EVERYTHING IN EVALUATION MODE
            self.network.eval()
            self.network.init_predictor.eval()
            self.network.denoiser.eval()
            
            #INIT METRIC DICTIONARY
            test_results = OrderedDict()
            test_results["psnr"] = []
            test_results["ssim"] = []
            test_results["fmeasure"] = []
            test_results["pseudof"] = []
            test_results["drd"] = []

            tq = tqdm(self.dataloader_test)
            iteration = 0
            #FOR IMAGE IN VALIDATION DATASET
            for img, gt, _ in tq:
                tq.set_description(f'VALIDATION PHASE {current_iteration} Iteration {iteration} / {len(self.dataloader_test.dataset)}')
                iteration += 1
                
                #IF NATIVE RESOLUTION SPLIT IMAGES IN MULTIPLE SUBIMAGES
                if self.native_resolution == 'True':
                    temp = img
                    img = crop_concat(img)
                #INIT RANDOM NOISE
                noisyImage = torch.randn_like(img).to(self.device)
                
                #FIRST INITIAL PREDICTION
                init_predict = self.network.init_predictor(img.to(self.device))

                #REFINE RESIDUAL IMAGE USING DPM SOLVER OR DDIM
                if self.DPM_SOLVER == 'True':   
                    #DPM SOLVER BRANCH
                    sampledImgs = dpm_solver(self.schedule.get_betas(), self.network.denoiser,
                                             noisyImage, self.DPM_STEP, init_predict,model_kwargs={})
                else:
                    #DDIM BRANCH
                    sampledImgs = self.diffusion(noisyImage.cuda(), init_predict, self.pre_ori)
                
                #COMPUTE FINAL IMAGES
                finalImgs = (sampledImgs + init_predict)
                
                #IF NATIVE RESOLUTION RECONSTRUCT FINAL IMAGES FROM MULTIPLE SUBIMAGES
                if self.native_resolution == 'True':
                    finalImgs = crop_concat_back(temp, finalImgs)
                    init_predict = crop_concat_back(temp, init_predict)
                    sampledImgs = crop_concat_back(temp, sampledImgs)
                    img = temp

                finalImgs = torch.clamp(finalImgs,0,1)
                
                #METRIC COMPUTATION AND LOGGING
                fmeasure, pfmeasure, psnr, drd = compute_metrics_DIBCO(finalImgs[0].cpu(),gt[0].cpu())
                test_results["psnr"].append(psnr)
                test_results["fmeasure"].append(fmeasure)
                test_results["pseudof"].append(pfmeasure)
                test_results["drd"].append(drd)
                ssim = self.ssim(gt.to(self.device),finalImgs.to(self.device)).item()
                test_results["ssim"].append(ssim)


            #AVERAGE METRICS COMPUTATION AND LOGGING
            self.logger.info(
                "----Average results for {}. Iteration {} ----\n".format(
                "Blur dataset validation", current_iteration
            ))
            log_dict = {}
            ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
            self.logger.info( "----Average PSNR\t: {:.6f}\n".format(ave_psnr) )
            ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
            self.logger.info( "----Average SSIM\t: {:.6f}\n".format(ave_ssim) )
            ave_fmeasure =   sum(test_results["fmeasure"]) / len(test_results["fmeasure"]) 
            self.logger.info( "----Average FMeasure\t: {:.6f}\n".format(ave_fmeasure) )
            ave_pfmeasure =    sum(test_results["pseudof"]) / len(test_results["pseudof"]) 
            self.logger.info( "----Average PFMeasure\t: {:.6f}\n".format(ave_pfmeasure) )
            ave_drd =    sum(test_results["drd"]) / len(test_results["drd"])
            self.logger.info( "----Average DRD\t: {:.6f}\n".format(ave_drd) )

            if self.wandb:
                log_dict['psnr'] = ave_psnr
                log_dict['ssim'] = ave_ssim
                log_dict['fmeasure'] = ave_fmeasure
                log_dict['pfmeasure'] = ave_pfmeasure
                log_dict['drd'] = ave_drd
                wandb.log(log_dict,step=current_iteration)
  
            if not os.path.exists(self.weight_save_path):
                os.makedirs(self.weight_save_path)

            #SAVE BEST MODELS WRT PSNR, FMEASURE AND Pseudo-FMEASURE
            if ave_psnr > self.bestPSNR:
                self.bestPSNR=ave_psnr
                to_save = {
                        'iteration': current_iteration,
                        'model_state_dict': self.network.init_predictor.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'bestPSNR': self.bestPSNR,
                        'bestFmeasure': self.bestFmeasure if self.bestFmeasure > ave_fmeasure else ave_fmeasure,
                        'bestPFmeasure': self.bestPFmeasure if self.bestPFmeasure  < ave_pfmeasure else ave_pfmeasure,
                }
                torch.save(to_save, #self.network.init_predictor.state_dict(),
                    os.path.join(self.weight_save_path, f'BEST_PSNR_model_init.pth'))
                
                to_save = {
                        'iteration': current_iteration,
                        'model_state_dict': self.network.denoiser.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'bestPSNR': self.bestPSNR,
                        'bestFmeasure': self.bestFmeasure if self.bestFmeasure > ave_fmeasure else ave_fmeasure,
                        'bestPFmeasure': self.bestPFmeasure if self.bestPFmeasure  < ave_pfmeasure else ave_pfmeasure,
                }
                torch.save(to_save,#self.network.denoiser.state_dict(),
                    os.path.join(self.weight_save_path, f'BEST_PSNR_model_denoiser.pth'))

            if ave_fmeasure > self.bestFmeasure:
                self.bestFmeasure = ave_fmeasure
                to_save = {
                        'iteration': current_iteration,
                        'model_state_dict': self.network.init_predictor.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'bestPSNR': self.bestPSNR if self.bestPSNR > ave_psnr else ave_psnr,
                        'bestFmeasure': self.bestFmeasure,
                        'bestPFmeasure': self.bestPFmeasure if self.bestPFmeasure  < ave_pfmeasure else ave_pfmeasure,
                }
                torch.save(to_save, #self.network.init_predictor.state_dict(),
                    os.path.join(self.weight_save_path, f'BEST_Fmeasure_model_init.pth'))
                to_save = {
                        'iteration': current_iteration,
                        'model_state_dict': self.network.denoiser.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'bestPSNR': self.bestPSNR if self.bestPSNR > ave_psnr else ave_psnr,
                        'bestFmeasure': self.bestFmeasure,
                        'bestPFmeasure': self.bestPFmeasure if self.bestPFmeasure  < ave_pfmeasure else ave_pfmeasure,
                }                
                torch.save(to_save,#self.network.denoiser.state_dict(),
                    os.path.join(self.weight_save_path, f'BEST_Fmeasure_model_denoiser.pth'))

            if ave_pfmeasure < self.bestPFmeasure:
                self.bestPFmeasure = ave_pfmeasure
                to_save = {
                        'iteration': current_iteration,
                        'model_state_dict': self.network.init_predictor.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'bestPSNR': self.bestPSNR if self.bestPSNR > ave_psnr else ave_psnr,
                        'bestFmeasure': self.bestFmeasure if self.bestFmeasure > ave_fmeasure else ave_fmeasure,
                        'bestPFmeasure': self.bestPFmeasure,
                }
                torch.save(to_save,# self.network.init_predictor.state_dict(),
                    os.path.join(self.weight_save_path, f'BEST_PFmeasure_model_init.pth'))
                to_save = {
                        'iteration': current_iteration,
                        'model_state_dict': self.network.denoiser.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'bestPSNR': self.bestPSNR if self.bestPSNR > ave_psnr else ave_psnr,
                        'bestFmeasure': self.bestFmeasure if self.bestFmeasure > ave_fmeasure else ave_fmeasure,
                        'bestPFmeasure': self.bestPFmeasure,
                }
                torch.save(to_save,#self.network.denoiser.state_dict(),
                    os.path.join(self.weight_save_path, f'BEST_PFmeasure_model_denoiser.pth'))



    # MAIN TRAIN FUNCTION
    def train(self):

        optimizer = self.optimizer
        iteration = self.continue_training_steps

        print('Starting Training', f"Step is {self.num_timesteps}")
        # UNTIL MAX ITERATION LIMIT IS REACHED
        while iteration < self.iteration_max:
            tq = tqdm(self.dataloader_train)

            for img, gt, _ in tq:
                tq.set_description(f'Iteration {iteration} / {self.iteration_max}')
                self.network.train()
                self.network.init_predictor.train()
                self.network.denoiser.train()
                
                optimizer.zero_grad()
                #SELECT TIMESTEP VECTOR T
                t = torch.randint(0, self.num_timesteps, (img.shape[0],)).long().to(self.device)
                
                #PASS IMAGES AND T THROUGH THE NETWORK
                init_predict, noise_pred, noisy_image, noise_ref = self.network(gt.to(self.device), img.to(self.device),
                                                                                t, self.diffusion)
                
                if self.pre_ori == 'True':
                    if self.high_low_freq == 'True':
                        residual_high = self.high_filter(gt.to(self.device) - init_predict)
                        ddpm_loss = 2*self.loss(self.high_filter(noise_pred), residual_high) + self.loss(noise_pred, gt.to(self.device) - init_predict)
                    else:
                        ddpm_loss = self.loss(noise_pred, gt.to(self.device) - init_predict)
                else:
                    ddpm_loss = self.loss(noise_pred, noise_ref.to(self.device))
                    
                if self.high_low_freq == 'True':
                    low_high_loss = self.loss(init_predict, gt.to(self.device))
                    low_freq_loss = self.loss(init_predict - self.high_filter(init_predict), gt.to(self.device) - self.high_filter(gt.to(self.device)))
                    pixel_loss = low_high_loss + 2*low_freq_loss
                else:
                    pixel_loss = self.loss(init_predict, gt.to(self.device))

                loss = ddpm_loss + self.beta_loss * (pixel_loss) / self.num_timesteps
                loss.backward()
                optimizer.step()
                
                if self.high_low_freq == 'True':
                    tq.set_postfix(loss=loss.item(), high_freq_ddpm_loss=ddpm_loss.item(), low_freq_pixel_loss=low_freq_loss.item(), pixel_loss=low_high_loss.item())
                else:
                    tq.set_postfix(loss=loss.item(), ddpm_loss=ddpm_loss.item(), pixel_loss=pixel_loss.item())
                if iteration % 1000 == 0:
                    if self.wandb:
                        wandb.log({'Loss':loss},step=iteration)
                    if not os.path.exists(self.save_img_path):
                        os.makedirs(self.save_img_path)
                    img_save = torch.cat([img, gt, init_predict.cpu()], dim=3)
                    if self.pre_ori == 'True':
                        if self.high_low_freq == 'True':
                            img_save = torch.cat([img, gt, init_predict.cpu(), noise_pred.cpu() + self.high_filter(init_predict).cpu(), noise_pred.cpu() + init_predict.cpu()], dim=3)
                        else:
                            img_save = torch.cat([img, gt, init_predict.cpu(), noise_pred.cpu() + init_predict.cpu()], dim=3)
                    save_image(img_save, os.path.join(
                        self.save_img_path, f"{iteration}.png"), nrow=4)
                
                iteration += 1
                
                #VALIDATION PHASE
                if iteration % self.validate_every == 0  :
                    self.validate(iteration)
                
                
                if self.EMA_or_not == 'True':
                    if iteration % self.ema_every == 0 and iteration > self.start_ema:
                        print('EMA update')
                        self.EMA.update_model_average(self.ema_model, self.network)

                if iteration % self.save_model_every == 0:
                    print('Saving models')
                    if not os.path.exists(self.weight_save_path):
                        os.makedirs(self.weight_save_path)
                    to_save = {
                        'iteration': iteration,
                        'model_state_dict': self.network.init_predictor.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'bestPSNR': self.bestPSNR ,
                        'bestFmeasure': self.bestFmeasure ,
                        'bestPFmeasure': self.bestPFmeasure,
                    }
                    torch.save(to_save,
                               os.path.join(self.weight_save_path, f'model_init_{iteration}.pth'))
                    to_save = {
                        'iteration': iteration,
                        'model_state_dict': self.network.denoiser.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'bestPSNR': self.bestPSNR ,
                        'bestFmeasure': self.bestFmeasure ,
                        'bestPFmeasure': self.bestPFmeasure,
                    }
                    torch.save(to_save,
                               os.path.join(self.weight_save_path, f'model_denoiser_{iteration}.pth'))



def dpm_solver(betas, model, x_T, steps, condition, model_kwargs):
    # You need to firstly define your model and the extra inputs of your model,
    # And initialize an `x_T` from the standard normal distribution.
    # `model` has the format: model(x_t, t_input, **model_kwargs).
    # If your model has no extra inputs, just let model_kwargs = {}.

    # If you use discrete-time DPMs, you need to further define the
    # beta arrays for the noise schedule.

    # model = ....
    # model_kwargs = {...}
    # x_T = ...
    # betas = ....

    # 1. Define the noise schedule.
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

    # 2. Convert your discrete-time `model` to the continuous-time
    # noise prediction model. Here is an example for a diffusion model
    # `model` with the noise prediction type ("noise") .
    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type="x_start",  # or "x_start" or "v" or "score"
        model_kwargs=model_kwargs,
        guidance_type="classifier-free",
        condition=condition
    )

    # 3. Define dpm-solver and sample by singlestep DPM-Solver.
    # (We recommend singlestep DPM-Solver for unconditional sampling)
    # You can adjust the `steps` to balance the computation
    # costs and the sample quality.
    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                            correcting_x0_fn="dynamic_thresholding")
    # Can also try
    # dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

    # You can use steps = 10, 12, 15, 20, 25, 50, 100.
    # Empirically, we find that steps in [10, 20] can generate quite good samples.
    # And steps = 20 can almost converge.
    x_sample = dpm_solver.sample(
        x_T,
        steps=steps,
        order=1,
        skip_type="time_uniform",
        method="singlestep",
    )
    return x_sample
