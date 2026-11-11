"""Zero-shot cloning from text, reference audio and its transcript."""
from inference import inference_parser, load_from_args


if __name__ == "__main__":
    parser = inference_parser(__doc__)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.set_defaults(output="clone.wav")
    args = parser.parse_args()
    tts = load_from_args(args)
    audio = tts.infer(args.text, prompt_wav=args.prompt_wav, prompt_text=args.prompt_text,
                      lang=args.lang, accent=args.accent, caption=args.caption)
    print(tts.save(args.output, audio))
