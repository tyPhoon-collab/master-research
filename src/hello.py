def main():
    import torch
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    prompt = "nekomimi maid"

    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    # Disable the NSFW filter by replacing the safety checker function
    def dummy_safety_checker(images, **kwargs):
        return images, [False] * len(images)

    pipeline.safety_checker = dummy_safety_checker

    # print(pipeline.scheduler.compatibles)
    # pipeline.scheduler = DDIMScheduler.from_config(
    #     pipeline.scheduler.config,
    # )
    pipeline = pipeline.to("cuda")
    generator = torch.Generator(device="cuda")
    image = pipeline(
        prompt,
        generator=generator,
        num_inference_steps=50,
    ).images[0]

    image.save("output/image/test.png")


if __name__ == "__main__":
    main()
