import argparse

import numpy as np
from tqdm import tqdm

import torch
from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import GreedyDecoder
from opts import add_decoder_args, add_inference_args
from utils import load_model
from loss import *
from reformer import *

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-npy', metavar='DIR',
                    help='path to validation manifest csv')
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for testing')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--save-output', default=None, help="Saves output of model from test to this file_path")
parser.add_argument('--ratio', type=float, default=1)
parser = add_decoder_args(parser)


def evaluate(device, model, decoder, audio_data=None, target=None):
  model.eval()
  total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
  output_data = []
  # print(npy)
  array = audio_data
  inputs = array
  inputs = inputs.to(device)
  input_sizes = torch.Tensor([inputs.size(3)])
  # print(inputs.shape)
  # print(input_sizes)
  out, output_sizes = model(inputs, input_sizes)
  decoded_output, _ = decoder.decode(out, output_sizes)

  target_strings = target

  transcript, reference = decoded_output[0][0], target_strings[0][0]
  wer_inst = decoder.wer(transcript, reference)
  cer_inst = decoder.cer(transcript, reference)
  # wer_inst = newWER(transcript, reference)
  # cer_inst = newCER(transcript, reference)
  return wer_inst, cer_inst


def TD(device, model, decoder, ratio=1, audio_data=None):
  model.eval()

  arr = audio_data
  inputs = arr
  inputs = inputs.to(device)
  input_sizes = torch.Tensor([inputs.size(3)])
  # print(inputs.shape)
  # print(input_sizes)
  out, output_sizes = model(inputs, input_sizes)
  decoded_output, _ = decoder.decode(out, output_sizes)

  arr_cut = arr[:, :, :, :round(arr.size(3) * ratio)]

  inputs = arr_cut
  inputs = inputs.to(device)
  input_sizes = torch.Tensor([inputs.size(3)])
  # # print(inputs.shape)
  # # print(input_sizes)
  out, output_sizes = model(inputs, input_sizes)
  decoded_output_cut, _ = decoder.decode(out, output_sizes)
  # decoded_output.append(decoded_output_cut[0])
  # print(decoded_output_cut)
  # print(decoded_output)

  if not decoded_output[0][0] or not decoded_output_cut[0][0]:
    return 1, 1
  wer_inst = newWER(decoded_output[0][0], decoded_output_cut[0][0])
  cer_inst = newCER(decoded_output[0][0], decoded_output_cut[0][0])
  # print("adv WER:", wer_inst)
  # print("adv CER:", cer_inst)
  # if wer_inst==0 or cer_inst==0:
  #   print(decoded_output_cut)
  #   print(decoded_output)
  return wer_inst, cer_inst


if __name__ == '__main__':
  args = parser.parse_args()
  torch.set_grad_enabled(False)
  device = torch.device("cuda" if args.cuda else "cpu")
  model = load_model(device, args.model_path, args.half)

  if args.decoder == "beam":
    from decoder import BeamCTCDecoder

    decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                             cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                             beam_width=args.beam_width, num_processes=args.lm_workers)
  elif args.decoder == "greedy":
    decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
  else:
    decoder = None
  target_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
  manifest_file = open(args.test_manifest)
  targets = manifest_file.readlines()
  # print([[targets[0].replace('\n','').split(',')[1].upper()]])

  threshold=0.3
  for i in range(10):
    threshold=i/10
    min_wer, min_cer = 1, 1
    werlist, cerlist = [0] * 11, [0] * 11
    # count = 0
    defense_wer,defense_cer=0,0
    orig_wer,orig_cer=0,0

    for batch_idx, c in enumerate(train_loader):
      c = c.to(DEVICE)
      orig_c = c
      c = c.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2)
      # print(c.shape)
      # recon = model(c)
      wer, cer = TD(device=device,
                    model=model,
                    decoder=decoder,
                    ratio=args.ratio,
                    audio_data=c)
      # if wer < min_wer and wer != 0:
      #   min_wer = wer
      # if cer < min_cer and cer != 0:
      #   min_cer = cer
      target = [[targets[batch_idx].replace('\n','').split(',')[1].upper()]]
      if cer > threshold:
        # count += 1
        c = reformer_model(orig_c)
        c = c.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2)
        wer,cer = evaluate(device, model, decoder, audio_data=c, target=target)
      else:
        wer,cer = evaluate(device, model, decoder, audio_data=c, target=target)
      # print(wer,cer)
      defense_wer+=wer
      defense_cer+=cer
      c = orig_c.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2)
      wer,cer = evaluate(device, model, decoder, audio_data=c, target=target)
      orig_wer+=wer
      orig_cer+=cer
      # werlist[int(round(wer * 10))] += 1
      # cerlist[int(round(cer * 10))] += 1

      torch.cuda.empty_cache()
    # print(min_wer,min_cer)
    # print(werlist,cerlist)
    # print(count)
    max_wer, max_cer = 0, 0
    werlist, cerlist = [0] * 11, [0] * 11
    print(threshold,'adv,defense',defense_wer/100,defense_cer/100)
    print(threshold,'adv,original',orig_wer/100,orig_cer/100)
    defense_wer,defense_cer=0,0
    orig_wer,orig_cer=0,0
    for batch_idx, c in enumerate(train_clean_loader):
      c = c.to(DEVICE)
      orig_c = c
      c = c.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2)
      # print(c.shape)
      # recon = model(c)

      wer, cer = TD(device=device,
                    model=model,
                    decoder=decoder,
                    ratio=args.ratio,
                    audio_data=c)
      # if wer > max_wer and wer != 1:
      #   max_wer = wer
      # if cer > max_cer and cer != 1:
      #   max_cer = cer
      target = [[targets[batch_idx].replace('\n','').split(',')[1].upper()]]
      if cer > threshold:
        # count += 1
        c = reformer_model(orig_c)
        c = c.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2)
        wer,cer = evaluate(device, model, decoder, audio_data=c, target=target)
      else:
        wer,cer = evaluate(device, model, decoder, audio_data=c, target=target)
      defense_wer+=wer
      defense_cer+=cer
      c = orig_c.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2)
      wer,cer = evaluate(device, model, decoder, audio_data=c, target=target)
      orig_wer+=wer
      orig_cer+=cer
      # werlist[int(round(wer * 10))] += 1
      # cerlist[int(round(cer * 10))] += 1
    print(threshold,'clean,defense',defense_wer/100,defense_cer/100)
    print(threshold,'clean,original',orig_wer/100,orig_cer/100)
    torch.cuda.empty_cache()