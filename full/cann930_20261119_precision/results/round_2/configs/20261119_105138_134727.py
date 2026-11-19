cli_args=dict(
    {'agent': None,
    'agent_api_key': None,
    'agent_dataset_path': None,
    'agent_deps': None,
    'agent_env': None,
    'agent_import_path': None,
    'agent_kwarg': None,
    'api_base': None,
    'api_key': None,
    'batch_size': 8,
    'cfg_time_str': '20261119_105138',
    'config': '/mnt/share/c00954197/Workspace/dsv41/precision_cann930/gpqa_config.py',
    'config_dir': 'configs',
    'custom_dataset_data_type': None,
    'custom_dataset_infer_method': None,
    'custom_dataset_meta_path': None,
    'custom_dataset_path': None,
    'dataset': None,
    'dataset_abbr': None,
    'datasets': None,
    'debug': False,
    'delete/__no_delete': None,
    'dir_time_str': '20261119_105138',
    'disable_verification': None,
    'dry_run': False,
    'dump_eval_details': True,
    'dump_extract_rate': False,
    'env_file': None,
    'environment': None,
    'exclude_task_name': None,
    'extra_docker_compose': None,
    'force_build/__no_force_build': None,
    'generation_kwargs': dict(
        chat_template_kwargs=dict(
            enable_thinking=True,
            reasoning_effort='low'),
        ignore_eos=False,
        seed=42,
        temperature=1.0,
        top_k=0,
        top_p=0.95),
    'host_ip': '177.42.23.1',
    'host_network': None,
    'host_port': 8000,
    'include_task_name': None,
    'max_num_workers': 1,
    'max_out_len': 65536,
    'max_retries': None,
    'max_workers_per_gpu': 1,
    'merge_ds': False,
    'mode': 'all',
    'model': None,
    'model_abbr': 'dsv41-bf16-cann930-gpqa',
    'model_name': 'dsv41-bf16-cann930',
    'models': None,
    'monitor_port': 0,
    'n_attempts': None,
    'n_concurrent': None,
    'n_tasks': None,
    'num_prompts': 198,
    'num_warmups': 0,
    'path': '/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16',
    'pressure': False,
    'pressure_time': 15,
    'purge_exception_cases': False,
    'quiet': None,
    'request_count': None,
    'request_rate': None,
    'response_anomaly': False,
    'response_anomaly_payload_retention': None,
    'retry': 1,
    'reuse': None,
    'search': False,
    'spec_decode': False,
    'stream': None,
    'string_config_input': None,
    'string_config_output': None,
    'summarizer': None,
    'timeout_multiplier': None,
    'tokenid_config_prefix_len': None,
    'tokenid_config_request_size': None,
    'trust_remote_code': None,
    'url': None,
    'use_timestamp': None,
    'work_dir': '/mnt/share/c00954197/Workspace/dsv41/precision_cann930/results/gpqa_20261119T105138Z_hWw1BM',
    'yes': None})
datasets=[
    dict(abbr='GPQA_diamond',
        eval_cfg=dict(
            evaluator=dict(
                type='ais_bench.benchmark.datasets.gpqa.GPQAEvaluator'),
            pred_postprocessor=dict(
                type='ais_bench.benchmark.datasets.gpqa.GPQA_Simple_Eval_postprocess')),
        infer_cfg=dict(
            inferencer=dict(
                type='ais_bench.benchmark.openicl.icl_inferencer.icl_gen_inferencer.GenInferencer'),
            prompt_template=dict(
                template=dict(
                    round=[
                        dict(prompt="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
                            role='HUMAN'),
                        ]),
                type='ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template.PromptTemplate'),
            retriever=dict(
                prompt_template=dict(
                    template=dict(
                        round=[
                            dict(prompt="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
                                role='HUMAN'),
                            ]),
                    type='ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template.PromptTemplate'),
                type='ais_bench.benchmark.openicl.icl_retriever.icl_zero_retriever.ZeroRetriever')),
        name='gpqa_diamond.csv',
        path='ais_bench/datasets/gpqa/',
        reader_cfg=dict(
            input_columns=[
                'question',
                'A',
                'B',
                'C',
                'D',
                ],
            output_column='answer',
            test_range='[:198]'),
        type='ais_bench.benchmark.datasets.gpqa.GPQADataset'),
    ]
eval=dict(
    partitioner=dict(
        out_dir='/mnt/share/c00954197/Workspace/dsv41/precision_cann930/results/gpqa_20261119T105138Z_hWw1BM/20261119_105138/results/',
        type='ais_bench.benchmark.partitioners.naive.NaivePartitioner'),
    runner=dict(
        debug=False,
        max_num_workers=1,
        max_workers_per_gpu=1,
        task=dict(
            cal_extract_rate=False,
            dump_details=True,
            type='ais_bench.benchmark.tasks.openicl_eval.OpenICLEvalTask'),
        type='ais_bench.benchmark.runners.local.LocalRunner'))
infer=dict(
    partitioner=dict(
        out_dir='/mnt/share/c00954197/Workspace/dsv41/precision_cann930/results/gpqa_20261119T105138Z_hWw1BM/20261119_105138/predictions/',
        type='ais_bench.benchmark.partitioners.naive.NaivePartitioner'),
    runner=dict(
        debug=False,
        max_num_workers=1,
        max_workers_per_gpu=1,
        task=dict(
            type='ais_bench.benchmark.tasks.openicl_api_infer.OpenICLApiInferTask'),
        type='ais_bench.benchmark.runners.local.LocalRunner'))
models=[
    dict(abbr='dsv41-bf16-cann930-gpqa',
        api_key='',
        attr='service',
        batch_size=8,
        generation_kwargs=dict(
            chat_template_kwargs=dict(
                enable_thinking=True,
                reasoning_effort='low'),
            ignore_eos=False,
            seed=42,
            temperature=1.0,
            top_k=0,
            top_p=0.95),
        host_ip='177.42.23.1',
        host_port=8000,
        max_out_len=65536,
        model='dsv41-bf16-cann930',
        path='/mnt/share/c00954197/model/deepseek-V4.1-Flash-1017-bf16',
        pred_postprocessor=dict(
            type='ais_bench.benchmark.utils.postprocess.model_postprocessors.extract_non_reasoning_content'),
        request_rate=0,
        request_timeout=360000,
        retry=1,
        stream=False,
        trust_remote_code=False,
        type='gpqa_api.GPQAAPIChat',
        url='',
        use_timestamp=False),
    ]
response_anomaly=dict(
    enabled=False,
    msprobe_config_path=None,
    payload_retention='anomalies',
    payload_storage=dict(
        compression='zstd',
        compression_level=3,
        format='jsonl',
        rows_per_shard=2000))
summarizer=dict(
    type='ais_bench.benchmark.summarizers.default.DefaultSummarizer')
work_dir='/mnt/share/c00954197/Workspace/dsv41/precision_cann930/results/gpqa_20261119T105138Z_hWw1BM/20261119_105138'