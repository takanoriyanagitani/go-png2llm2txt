package main

import (
	"context"
	"flag"
	"iter"
	"log"
	"os"

	oa "github.com/ollama/ollama/api"
	pt "github.com/takanoriyanagitani/go-png2llm2txt"
)

type config struct {
	model    string
	prompt   string
	system   string
	temp     float64
	seed     int
	noStream bool
	think    string
	limit    int64
}

func (c config) getThink() pt.Think {
	switch c.think {
	case "true":
		return pt.DoThink
	case "false":
		return pt.DontThink
	case "high":
		return pt.ThinkH
	case "medium":
		return pt.ThinkM
	case "low":
		return pt.ThinkL
	default:
		return nil
	}
}

func sub(ctx context.Context, cfg config) error {
	rdr := pt.RawImageReader{Reader: os.Stdin}

	rimg, err := rdr.ToRaw(cfg.limit)
	if nil != err {
		return err
	}

	png, err := rimg.ToPng()
	if nil != err {
		return err
	}

	opts := oa.Options{
		Temperature: float32(cfg.temp),
		Seed:        cfg.seed,
	}

	req := pt.Request{
		PngData: png,
		Think:   cfg.getThink(),
		Options: pt.Options{Options: opts},

		Model:  cfg.model,
		Prompt: cfg.prompt,
		System: cfg.system,

		NoStream: cfg.noStream,

		Seed: cfg.seed,
	}

	var greq oa.GenerateRequest = req.ToGenRequest()

	ocli, err := oa.ClientFromEnvironment()
	if nil != err {
		return err
	}

	cli := pt.Client{Client: ocli}

	var ires iter.Seq2[oa.GenerateResponse, error] = cli.GenerateStreaming(
		ctx,
		&greq,
	)

	for res, err := range ires {
		if nil != err {
			return err
		}

		generated := pt.Generated{GenerateResponse: res}

		err = generated.ToStdout()
		if nil != err {
			return err
		}
	}

	return nil
}

func main() {
	var cfg config
	flag.StringVar(&cfg.model, "model", "", "ollama model (required)")
	flag.StringVar(&cfg.prompt, "prompt", "explain the image", "prompt")
	flag.StringVar(&cfg.system, "system", "", "system prompt")
	flag.Float64Var(&cfg.temp, "temp", 0.0, "temperature")
	flag.IntVar(&cfg.seed, "seed", 0, "seed")
	flag.BoolVar(&cfg.noStream, "nostream", false, "no stream")
	flag.StringVar(&cfg.think, "think", "none", "think: none, true, false, high, medium, low")
	flag.Int64Var(&cfg.limit, "limit", 16777216, "image size limit")
	flag.Parse()

	if cfg.model == "" {
		flag.Usage()
		os.Exit(1)
	}

	err := sub(context.Background(), cfg)
	if nil != err {
		log.Printf("%v\n", err)
	}
}
