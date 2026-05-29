#!/bin/bash

set -e

export DASHSCOPE_API_KEY="sk-68a39855d0ec4d8ea23999d4d5ccd306"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export OPENAI_API_KEY="" # API KEY FOR OPENAI CHATGPT
export GOOGLE_API_KEY="" # API KEY FOR GOGOLE GEMINI

benchmark=vsibench
timezone="Asia/Shanghai"
output_path=logs/$(TZ="$timezone" date "+%Y%m%d")
num_processes=4
num_frames=16
launcher=accelerate
wandb_args=""
use_wandb_args=false
video_sampling_strategy="uniform"
video_sample_fps=""
video_input_mode=""
visual_input_mode="visual"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
answer_mode="restricted"
gen_kwargs=""
use_gen_kwargs=false
natural_gen_kwargs="max_new_tokens=256,temperature=0,top_p=1.0,num_beams=1,do_sample=false"
run_note=""

available_models="gemini_3_1_pro,gemini_3_1_flash_lite,gpt5_4,llava_one_vision_1_5_8b,llava_next_video_7b_qwen2,internvl3_5_2b,internvl3_5_8b,qwen3vl_8b,qwen3vl_32b,qwen2_5vl_72b_api,qwen3vl_235b_a22b_api,internvideo2_5_chat_8b"
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
    --num_frames)
        num_frames="$2"
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
    --timezone)
        timezone="$2"
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
    --video_sample_fps)
        video_sample_fps="$2"
        shift 2
        ;;
    --video_input_mode)
        video_input_mode="$2"
        shift 2
        ;;
    --visual_input_mode)
        visual_input_mode="$2"
        shift 2
        ;;
    --cuda_visible_devices)
        cuda_visible_devices="$2"
        shift 2
        ;;
    --answer_mode)
        answer_mode="$2"
        shift 2
        ;;
    --gen_kwargs)
        gen_kwargs="$2"
        use_gen_kwargs=true
        shift 2
        ;;
    --run_note|--note)
        run_note="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

case "$visual_input_mode" in
"visual"|"none")
    ;;
*)
    echo "Unsupported --visual_input_mode: $visual_input_mode (expected: visual or none)"
    exit 1
    ;;
esac

case "$answer_mode" in
"restricted"|"natural")
    ;;
*)
    echo "Unsupported --answer_mode: $answer_mode (expected: restricted or natural)"
    exit 1
    ;;
esac

export VSI_VISUAL_INPUT_MODE="$visual_input_mode"
export VSI_ANSWER_MODE="$answer_mode"
export VSI_RUN_NOTE="$run_note"
requested_num_processes="$num_processes"
requested_launcher="$launcher"

export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    gpu_count=${#devices[@]}
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (gpu_count=$gpu_count)"

if [ "${#models[@]}" -eq 1 ] && [ "${models[0]}" = "all" ]; then
    IFS=',' read -r -a models <<<"$available_models"
fi

quote_arg() {
    printf "%q" "$1"
}

for model in "${models[@]}"; do
    echo "Start evaluating $model..."
    num_processes="$requested_num_processes"
    launcher="$requested_launcher"

    case "$model" in
    "gemini_3_1_pro")
        model_family="gemini3_1_pro"
        model="gemini_3_1_pro_${num_frames}f"
        model_args="model_version=gemini-3.1-pro-preview,modality=video,max_frames_num=$num_frames"
        num_processes=1
        launcher=python
        ;;
    "gemini_3_1_flash_lite")
        model_family="gemini3_1_flash_lite"
        model="gemini_3_1_flash_lite_${num_frames}f"
        model_args="model_version=gemini-3.1-flash-lite,modality=video,max_frames_num=$num_frames"
        num_processes=1
        launcher=python
        ;;
    "gpt5_4")
        model_family="gpt5_4"
        model="gpt5_4_${num_frames}f"
        model_args="model_version=gpt-5.4,modality=video,max_frames_num=$num_frames,image_detail=auto"
        num_processes=1
        launcher=python
        ;;
    "llava_one_vision_1_5_8b")
        model_family="llava_onevision_1_5"
        model="llava_one_vision_1_5_8b_${num_frames}f"
        model_args="pretrained=~/.cache/modelscope/hub/models/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct,attn_implementation=flash_attention_2,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=$num_frames,max_pixels=602112"
        ;;
    "llava_one_vision_qwen2_72b_ov")
        model_family="llava_onevision"
        model="llava_one_vision_qwen2_72b_ov_${num_frames}f"
        model_args="pretrained=lmms-lab/llava-onevision-qwen2-72b-ov-sft,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=$num_frames,device_map=auto"
        num_processes=1
        ;;
    "llava_next_video_7b_qwen2")
        model_family="llava_vid"
        model="llava_next_video_7b_qwen2_${num_frames}f"
        model_args="pretrained=~/.cache/modelscope/hub/models/QwenCollection/LLaVA-NeXT-Video-7B-Qwen2/,video_decode_backend=decord,conv_template=qwen_1_5,max_frames_num=$num_frames"
        ;;
    "llava_next_video_72b_qwen2")
        model_family="llava_vid"
        model="llava_next_video_72b_qwen2_${num_frames}f"
        model_args="pretrained=lmms-lab/LLaVA-NeXT-Video-72B-Qwen2,video_decode_backend=decord,conv_template=qwen_1_5,max_frames_num=$num_frames,device_map=auto"
        num_processes=1
        ;;
    "internvl3_5_2b")
        model_family="internvl3_5"
        model="internvl3_5_2b_${num_frames}f"
        model_args="pretrained=~/.cache/modelscope/hub/models/OpenGVLab/InternVL3_5-2B,modality=video,max_frames_num=$num_frames,video_max_num=4"
        ;;
    "internvl3_5_8b")
        model_family="internvl3_5"
        model="internvl3_5_8b_${num_frames}f"
        # 8B 模型依然可以在多卡数据并行（num_processes=4）下良好运行
        model_args="pretrained=~/.cache/modelscope/hub/models/OpenGVLab/InternVL3_5-8B,modality=video,max_frames_num=$num_frames,video_max_num=4"
        ;;
    "qwen3vl_8b")
        model_family="qwen3vl"
        model="qwen3vl_8b_${num_frames}f"
        model_args="pretrained=~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct,modality=video,max_frames_num=$num_frames"
        ;;
    "qwen3vl_32b")
        model_family="qwen3vl_32b"
        model="qwen3vl_32b_${num_frames}f"
        model_args="pretrained=~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-32B-Instruct,modality=video,max_frames_num=$num_frames,device_map=auto"
        num_processes=1
        ;;
    "qwen2_5vl_72b_api")
        model_family="qwen2_5vl_72b_api"
        model="qwen2_5vl_72b_api_${num_frames}f"
        model_args="model_version=qwen2.5-vl-72b-instruct,modality=video,max_frames_num=$num_frames"
        num_processes=1
        launcher=python
        ;;
    "qwen3vl_235b_a22b_api")
        model_family="qwen3vl_235b_a22b_api"
        model="qwen3vl_235b_a22b_api_${num_frames}f"
        model_args="model_version=qwen3-vl-235b-a22b-instruct,modality=video,max_frames_num=$num_frames"
        num_processes=1
        launcher=python
        ;;
    "internvideo2_5_chat_8b")
        model_family="internvideo2_5_chat_8b"
        model="internvideo2_5_chat_8b_${num_frames}f"
        model_args="pretrained=~/.cache/modelscope/hub/models/OpenGVLab/InternVideo2_5_Chat_8B,modality=video,max_frames_num=$num_frames,device_map=auto"
        ;;
    *)
        echo "Unknown model: $model"
        exit 1
        ;;
    esac

    if [ "$visual_input_mode" = "none" ]; then
        model_args="$model_args,visual_input_mode=none"
        model="${model}_blind"
    else
        # Add sampling strategy into model_args
        if [ -n "$video_sample_fps" ]; then
            model_args="$model_args,video_sampling_strategy=fps,video_sample_fps=$video_sample_fps"
        elif [ "$video_sampling_strategy" = "specific" ]; then
            model_args="$model_args,video_sampling_strategy=specific,keyframe_mapping_path=data/keyframe_mapping.json"
        elif [ "$video_sampling_strategy" != "uniform" ]; then
            model_args="$model_args,video_sampling_strategy=$video_sampling_strategy"
        fi
        if [ -n "$video_input_mode" ]; then
            case "$model_family" in
            "qwen2_5vl_72b_api"|"qwen3vl_235b_a22b_api")
                model_args="$model_args,video_input_mode=$video_input_mode"
                model="${model}_${video_input_mode}"
                ;;
            *)
                echo "Warning: --video_input_mode is only supported by Qwen API adapters; ignoring for $model_family"
                ;;
            esac
        fi
    fi

    if [ "$answer_mode" = "natural" ]; then
        model="${model}_natural"
        if [ "$use_gen_kwargs" = false ]; then
            gen_kwargs="$natural_gen_kwargs"
            use_gen_kwargs=true
        fi
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
        --output_path $output_path/$benchmark \
        --timezone $timezone"

    if [ "$use_wandb_args" = true ]; then
        evaluate_script="$evaluate_script \
        --wandb_args $wandb_args"
    fi

    if [ "$use_gen_kwargs" = true ]; then
        evaluate_script="$evaluate_script \
        --gen_kwargs $gen_kwargs"
    fi

    if [ -n "$run_note" ]; then
        run_note_arg=$(quote_arg "$run_note")
        evaluate_script="$evaluate_script \
        --run_note $run_note_arg"
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
