package controllers

import (
	"bytes"
	"encoding/base64"
	"github.com/gin-gonic/gin"
	"github.com/joeyave/statistics-project3/global"
	"github.com/joeyave/statistics-project3/helpers"
	"github.com/joeyave/statistics-project3/templates"
	"gonum.org/v1/plot/plotter"
	"image/color"
	"math"
	"net/http"
)

const alpha = 0.05

func IdentifyNormDistribution(c *gin.Context) {

	x := global.CopyX()
	y := global.CopyY()

	var arr []map[string]interface{}
	var stats []*templates.Stat
	var statWithU *templates.StatWithU

	if global.DataType() == "dep" {

		var diff []float64
		for i := range x {
			diff = append(diff, x[i]-y[i])
		}

		params, paperImage, eCDFImage, diffKolmogorovStat, characteristics := identify(diff)

		arr = append(arr, map[string]interface{}{
			"Header":              "diff data",
			"Params":              params,
			"PaperImage":          paperImage,
			"eCDFImage":           eCDFImage,
			"IsNormStat":          diffKolmogorovStat,
			"StatCharacteristics": characteristics,
		})

		if diffKolmogorovStat.IsNorm {
			t, p := helpers.PairedTTest(x, y)
			stats = append(stats, &templates.Stat{
				Name:   "t (Paired t-test)",
				Val:    t,
				P:      p,
				IsNorm: p >= alpha,
				Alpha:  alpha,
			})
		}

	} else {

		params, paperImage, eCDFImage, xKolmogorovStat, characteristics := identify(x)

		arr = append(arr, map[string]interface{}{
			"Header":              "x data",
			"Params":              params,
			"PaperImage":          paperImage,
			"eCDFImage":           eCDFImage,
			"IsNormStat":          xKolmogorovStat,
			"StatCharacteristics": characteristics,
		})

		params, paperImage, eCDFImage, yKolmogorovStat, characteristics := identify(y)

		arr = append(arr, map[string]interface{}{
			"Header":              "y data",
			"Params":              params,
			"PaperImage":          paperImage,
			"eCDFImage":           eCDFImage,
			"IsNormStat":          yKolmogorovStat,
			"StatCharacteristics": characteristics,
		})

		if xKolmogorovStat.IsNorm && yKolmogorovStat.IsNorm {

			f, p := helpers.FTest(x, y)
			fTestStat := &templates.Stat{
				Name:   "f (F test)",
				Val:    f,
				P:      p,
				IsNorm: p >= alpha,
				Alpha:  alpha,
			}
			stats = append(stats, fTestStat)

			if fTestStat.IsNorm {
				t, p := helpers.TwoSampleTTest(x, y)
				stats = append(stats, &templates.Stat{
					Name:   "t (Two sample t-test)",
					Val:    t,
					P:      p,
					IsNorm: p >= alpha,
					Alpha:  alpha,
				})
			}

		} else {
			U := helpers.VanDerWaerdenTest(x, y)

			statWithU = &templates.StatWithU{
				Name:   "Van Der Waerden test",
				UAbs:   math.Abs(U),
				U:      helpers.QuantileU(1 - alpha/2),
				IsNorm: math.Abs(U) <= helpers.QuantileU(1-alpha/2),
				Alpha:  alpha,
			}
		}
	}

	c.HTML(http.StatusOK, "distribution.tmpl", map[string]interface{}{
		"FileName":  global.FileName(),
		"Arr":       arr,
		"Stats":     stats,
		"StatWithU": statWithU,
	})
}

func identify(data []float64) (params []*templates.Param, paperImage string, eCDFImage string, isNormStat *templates.Stat, characteristics []*templates.StatCharacteristic) {

	mu := helpers.NormMuMLE(data)
	sigmaSquared := helpers.NormSigmaSquaredMLE(data)

	params = append(params,
		&templates.Param{
			Name: "mu",
			Val:  mu,
		},
		&templates.Param{
			Name: "sigma^2",
			Val:  sigmaSquared,
		},
	)

	p := helpers.DistributionIdentificationPlot(func(x_i float64) float64 {
		y := helpers.QuantileU(helpers.EmpiricalCDF(data)(x_i))
		return y
	}, data)

	line := plotter.NewFunction(func(x_i float64) float64 {
		y := helpers.QuantileU(helpers.NormCDF(mu, math.Sqrt(sigmaSquared))(x_i))
		return y
	})
	line.Color = color.RGBA{G: 255, A: 255}

	p.Add(line)

	writer, err := p.WriterTo(helpers.PlotWidth, helpers.PlotHeight, "svg")
	if err != nil {
		return nil, "", "", nil, nil
	}

	buf := new(bytes.Buffer)
	_, err = writer.WriteTo(buf)
	if err != nil {
		return nil, "", "", nil, nil
	}

	paperImage = base64.StdEncoding.EncodeToString(buf.Bytes())

	p = helpers.PlotEmpiricalCDF(data)
	cdf := plotter.NewFunction(helpers.NormCDF(mu, math.Sqrt(sigmaSquared)))
	cdf.Color = color.RGBA{G: 255, A: 255}
	cdf.Width = 2
	p.Add(cdf)

	writer, err = p.WriterTo(helpers.PlotWidth, helpers.PlotHeight, "svg")
	if err != nil {
		return nil, "", "", nil, nil
	}

	buf = new(bytes.Buffer)
	_, err = writer.WriteTo(buf)
	if err != nil {
		return nil, "", "", nil, nil
	}

	eCDFImage = base64.StdEncoding.EncodeToString(buf.Bytes())

	z := helpers.KolmogorovZ(data, helpers.NormCDF(mu, math.Sqrt(sigmaSquared)))
	k := helpers.KolmogorovFunction(z, data)

	isNormStat = &templates.Stat{
		Name:   "z (Kolmogorov stat)",
		Val:    z,
		P:      1 - k,
		Alpha:  alpha,
		IsNorm: (1 - k) >= alpha,
	}

	mean := helpers.Mean(data)
	from, to := helpers.MeanConfidenceInterval(alpha, data)

	characteristics = append(characteristics, &templates.StatCharacteristic{
		Name: "mean",
		Val:  mean,
		From: from,
		To:   to,
	})

	median := helpers.Median(data)
	from, to = helpers.MedianConfidenceInterval(alpha, data)

	characteristics = append(characteristics, &templates.StatCharacteristic{
		Name: "median",
		Val:  median,
		From: from,
		To:   to,
	})

	//return map[string]interface{}{
	//	"Name":              name,
	//	"Image":             paperImage,
	//	"EmpiricalCDFImage": eCDFImage,
	//
	//	"Characteristics":     []*templates.StatCharacteristic{&muCharacteristic, &sigmaSquaredCharacteristic},
	//	"StatCharacteristics": characteristics,
	//
	//	"Z":      z,
	//	"P":      1 - k,
	//	"Alpha":  alpha,
	//	"IsNorm": (1 - k) >= alpha,
	//}

	return params, paperImage, eCDFImage, isNormStat, characteristics
}
