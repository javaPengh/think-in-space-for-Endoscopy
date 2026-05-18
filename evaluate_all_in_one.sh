#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=1,2,3,4
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    gpu_count=${#devices[@]}
fi

export OPENAI_API_KEY="" # API KEY FOR OPENAI CHATGPT
export GOOGLE_API_KEY="" # API KEY FOR GOGOLE GEMINI

benchmark=vsibench
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
num_processes=4
num_frames=16
launcher=accelerate
wandb_args=""
use_wandb_args=false
video_sampling_strategy="uniform"

available_models="llava_one_vision_qwen2_0p5b_ov_32f,llava_one_vision_qwen2_7b_ov_32f,llava_next_video_7b_qwen2_32f,llama3_vila1p5_8b_32f,llama3_longvila_8b_128frames_32f,longva_7b_32f,internvl2_2b_8f,internvl2_8b_8f"
IFS=',' read -r -a models <<<"$available_models"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --benchmark)
        benchmark="$2"
        shift 2
        ;;
    --num_processes)
        num_processes="$2"
        shift 2
        ;;
    --model)
        IFS=',' read -r -a models <<<"$2"
        shift 2
        ;;
    --output_path)
        output_path="$2"
        shift 2
        ;;
    --limit)
        limit="$2"
        shift 2
        ;;
    --wandb_args)
        use_wandb_args=true
        wandb_args="$2"
        shift 2
        ;;
    --video_sampling_strategy)
        video_sampling_strategy="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

if [ "${#models[@]}" -eq 1 ] && [ "${models[0]}" = "all" ]; then
    IFS=',' read -r -a models <<<"$available_models"
fi

for model in "${models[@]}"; do
    echo "Start evaluating $model..."

    case "$model" in
    "gemini_1p5_flash")
        model_family="gemini_api"
        model_args="model_version=gemini-1.5-flash,modality=video"
        ;;
    "gemini_1p5_pro_002")
        model_family="gemini_api"
        model_args="model_version=gemini-1.5-pro,modality=video"
        ;;
    "gemini_2p0_flash_exp")
        model_family="gemini_api"
        model_args="model_version=gemini-2.0-flash-exp,modality=video"
        ;;
    "gpt_4o_2024_08_06_f16")
        model_family="gpt4v"
        model_args="model_version=gpt-4o-2024-08-06,modality=video,max_frames_num=16"
        ;;
    "llava_one_vision_qwen2_0p5b_ov_32f")
        model_family="llava_onevision"
        model="llava_one_vision_qwen2_0p5b_ov_${num_frames}f"
        model_args="pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=$num_frames"
        ;;
    "llava_one_vision_qwen2_7b_ov_32f")
        model_family="llava_onevision"
        model="llava_one_vision_qwen2_7b_ov_${num_frames}f"
        model_args="pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=$num_frames"
        ;;
    "llava_one_vision_1_5_8b_32f")
        model_family="llava_onevision_1_5"
        model="llava_one_vision_1_5_8b_${num_frames}f"
        model_args="pretrained=~/.cache/modelscope/hub/models/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct,attn_implementation=flash_attention_2,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=$num_frames,max_pixels=602112,device_map=auto"
        ;;
    "llava_one_vision_qwen2_72b_ov_32f")
        model_family="llava_onevision"
        model_args="pretrained=lmms-lab/llava-onevision-qwen2-72b-ov-sft,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32,device_map=auto"
        num_processes=1
        ;;
    "llava_next_video_7b_qwen2_32f")
        model_family="llava_vid"
        model="llava_next_video_7b_qwen2_${num_frames}f"
        model_args="pretrained=~/.cache/modelscope/hub/models/QwenCollection/LLaVA-NeXT-Video-7B-Qwen2/,video_decode_backend=decord,conv_template=qwen_1_5,max_frames_num=$num_frames"
        ;;
    "llava_next_video_72b_qwen2_32f")
        model_family="llava_vid"
        model_args="pretrained=lmms-lab/LLaVA-NeXT-Video-72B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,max_frames_num=32,device_map=auto"
        num_processes=1
        ;;
    "internvl3_5_2b_32f")
        model_family="internvl3_5"
        model="internvl3_5_2b_${num_frames}f"
        model_args="pretrained=~/.cache/modelscope/hub/models/OpenGVLab/InternVL3_5-2B,modality=video,max_frames_num=$num_frames"
        ;;
    "internvl3_5_8b_32f")
        model_family="internvl3_5"
        model="internvl3_5_8b_${num_frames}f"
        # 8B 模型依然可以在多卡数据并行（num_processes=4）下良好运行
        model_args="pretrained=~/.cache/modelscope/hub/models/OpenGVLab/InternVL3_5-8B,modality=video,max_frames_num=$num_frames"
        ;;
    "qwen3vl_8b_32f")
        model_family="qwen3vl"
        model="qwen3vl_8b_${num_frames}f"
        model_args="pretrained=~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct,modality=video,max_frames_num=$num_frames"
        ;;
    "qwen3vl_32b_32f")
        model_family="qwen3vl_32b"
        model="qwen3vl_32b_${num_frames}f"
        model_args="pretrained=~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-32B-Instruct,modality=video,max_frames_num=$num_frames,device_map=auto"
        num_processes=1
        ;;
    "internvideo2_5_chat_8b_32f")
        model_family="internvideo2_5_chat_8b"
        model="internvideo2_5_chat_8b_${num_frames}f"
        model_args="pretrained=~/.cache/modelscope/hub/models/OpenGVLab/InternVideo2_5_Chat_8B,modality=video,max_frames_num=$num_frames,device_map=auto"
        ;;
    *)
        echo "Unknown model: $model"
        exit 1
        ;;
    esac

    # Add sampling strategy into model_args
    if [ "$video_sampling_strategy" = "specific" ]; then
        model_args="$model_args,video_sampling_strategy=specific,keyframe_mapping_path=data/keyframe_mapping.json"
    fi

    if [ "$launcher" = "python" ]; then
        export LMMS_EVAL_LAUNCHER="python"
        evaluate_script="python \
            "
    elif [ "$launcher" = "accelerate" ]; then
        export LMMS_EVAL_LAUNCHER="accelerate"
        evaluate_script="accelerate launch \
            --num_processes=$num_processes \
            "
    fi

    evaluate_script="$evaluate_script -m lmms_eval \
        --model $model_family \
        --model_args $model_args \
        --tasks $benchmark \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $model \
        --output_path $output_path/$benchmark"

    if [ "$use_wandb_args" = true ]; then
        evaluate_script="$evaluate_script \
        --wandb_args $wandb_args"
    fi

    evaluate_script="$evaluate_script \
        "

    if [ -n "$limit" ]; then
        evaluate_script="$evaluate_script \
            --limit $limit \
        "
    fi
    echo $evaluate_script
    eval $evaluate_script
done
