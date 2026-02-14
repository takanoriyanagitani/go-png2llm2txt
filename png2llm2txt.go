package png2llm2txt

import (
	"bytes"
	"context"
	"errors"
	"image/png"
	"io"
	"iter"
	"os"

	oa "github.com/ollama/ollama/api"
)

var (
	ErrCancel error = errors.New("iterator canceled")
)

type Client struct{ *oa.Client }

func (c Client) GenerateStreaming(
	ctx context.Context,
	req *oa.GenerateRequest,
) iter.Seq2[oa.GenerateResponse, error] {
	return func(yield func(oa.GenerateResponse, error) bool) {
		err := c.Client.Generate(
			ctx,
			req,
			func(res oa.GenerateResponse) error {
				if res.Done {
					return nil
				}

				if !yield(res, nil) {
					return ErrCancel
				}
				return nil
			},
		)

		if nil != err {
			yield(oa.GenerateResponse{}, err)
		}
	}
}

type Generated struct{ oa.GenerateResponse }

func (g Generated) String() string {
	return g.GenerateResponse.Response
}

func (g Generated) ToWriter(wtr io.Writer) error {
	_, err := io.WriteString(wtr, g.String())
	return err
}

func (g Generated) ToStdout() error { return g.ToWriter(os.Stdout) }

type Think func() oa.ThinkValue

func DoThink() oa.ThinkValue {
	return oa.ThinkValue{Value: true}
}

func DontThink() oa.ThinkValue {
	return oa.ThinkValue{Value: false}
}

func ThinkH() oa.ThinkValue { return oa.ThinkValue{Value: "high"} }
func ThinkM() oa.ThinkValue { return oa.ThinkValue{Value: "medium"} }
func ThinkL() oa.ThinkValue { return oa.ThinkValue{Value: "low"} }

//nolint:gochecknoglobals
var ThinkDefault Think = nil

func (t Think) Value() *oa.ThinkValue {
	if nil == t {
		return nil
	}

	var tval oa.ThinkValue = t()
	return &tval
}

type PngData []byte

type RawImage []byte

func (r RawImage) ToPng() (PngData, error) {
	var raw []byte = r
	_, e := png.DecodeConfig(bytes.NewReader(raw))
	return PngData(r), e
}

type RawImageReader struct{ io.Reader }

func (r RawImageReader) ToRaw(limit int64) (RawImage, error) {
	limited := &io.LimitedReader{
		R: r.Reader,
		N: limit,
	}
	var buf bytes.Buffer
	_, err := io.Copy(&buf, limited)
	return buf.Bytes(), err
}

type Options struct{ oa.Options }

//nolint:cyclop
func (o Options) ToMap() map[string]any {
	msa := make(map[string]any)

	if o.NumKeep != 0 {
		msa["num_keep"] = o.NumKeep
	}

	if o.Seed != 0 {
		msa["seed"] = o.Seed
	}

	if o.NumPredict != 0 {
		msa["num_predict"] = o.NumPredict
	}

	if o.TopK != 0 {
		msa["top_k"] = o.TopK
	}

	if o.TopP != 0 {
		msa["top_p"] = o.TopP
	}

	if o.MinP != 0 {
		msa["min_p"] = o.MinP
	}

	if o.TypicalP != 0 {
		msa["typical_p"] = o.TypicalP
	}

	if o.RepeatLastN != 0 {
		msa["repeat_last_n"] = o.RepeatLastN
	}

	if o.Temperature != 0 {
		msa["temperature"] = o.Temperature
	}

	if o.RepeatPenalty != 0 {
		msa["repeat_penalty"] = o.RepeatPenalty
	}

	if o.PresencePenalty != 0 {
		msa["presence_penalty"] = o.PresencePenalty
	}

	if o.FrequencyPenalty != 0 {
		msa["frequency_penalty"] = o.FrequencyPenalty
	}

	if len(o.Stop) > 0 {
		msa["stop"] = o.Stop
	}

	return msa
}

type Request struct {
	PngData
	Think
	Options

	Model    string
	Prompt   string
	System   string
	Template string

	NoStream bool

	Seed int
}

func (q Request) Streaming() *bool {
	var doStream bool = !q.NoStream
	return &doStream
}

func (q Request) ImageData() oa.ImageData {
	var b []byte = q.PngData
	return oa.ImageData(b)
}

func (q Request) ThinkValue() *oa.ThinkValue {
	return q.Think.Value()
}

func (q Request) ToGenRequest() oa.GenerateRequest {
	var imgs []oa.ImageData
	var img oa.ImageData = q.ImageData()
	if nil != img {
		imgs = append(imgs, img)
	}
	return oa.GenerateRequest{
		Model:    q.Model,
		Prompt:   q.Prompt,
		System:   q.System,
		Template: q.Template,
		Stream:   q.Streaming(),
		Images:   imgs,
		Options:  q.Options.ToMap(),
		Think:    q.ThinkValue(),
	}
}
