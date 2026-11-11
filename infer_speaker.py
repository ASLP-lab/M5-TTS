"""Preset speaker synthesis from text and a trained speaker ID (default: 1)."""
from inference import inference_parser, load_from_args


if __name__ == "__main__":
    parser = inference_parser(__doc__)
    parser.add_argument("--speaker-id", type=int, default=1)
    parser.set_defaults(output="speaker.wav")
    args = parser.parse_args()
    tts = load_from_args(args)
    audio = tts.infer(args.text, speaker_id=args.speaker_id,
                      lang=args.lang, accent=args.accent, caption=args.caption)
    print(tts.save(args.output, audio))
