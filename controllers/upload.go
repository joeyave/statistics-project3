package controllers

import (
	"bufio"
	"bytes"
	"github.com/gin-gonic/gin"
	"github.com/joeyave/statistics-project3/global"
	"log"
	"net/http"
	"strconv"
	"strings"
)

func Upload(c *gin.Context) {

	dataType := c.PostForm("type")
	global.SetDataType(dataType)

	file, err := c.FormFile("file")
	if err != nil {
		return
	}

	openedFile, err := file.Open()
	if err != nil {
		return
	}
	defer openedFile.Close()

	buf := new(bytes.Buffer)
	buf.ReadFrom(openedFile)

	var xs, ys []float64

	scanner := bufio.NewScanner(buf)
	for scanner.Scan() {
		text := scanner.Text()
		vals := strings.Fields(text)

		if dataType == "dep" {
			x, err := strconv.ParseFloat(vals[0], 64)
			if err != nil {
				log.Fatalln(err)
			}
			xs = append(xs, x)

			y, err := strconv.ParseFloat(vals[1], 64)
			if err != nil {
				log.Fatalln(err)
			}
			ys = append(ys, y)
		} else {
			val, err := strconv.ParseFloat(vals[0], 64)
			if err != nil {
				log.Fatalln(err)
			}

			class, err := strconv.ParseFloat(vals[1], 64)
			if err != nil {
				log.Fatalln(err)
			}

			if class == 0 {
				xs = append(xs, val)
			} else {
				ys = append(ys, val)
			}
		}
	}

	global.SetX(xs)
	global.SetY(ys)

	global.SetFileName(file.Filename)

	c.HTML(http.StatusOK, "upload.tmpl", map[string]interface{}{
		"FileName": file.Filename,
		"X":        xs,
		"Y":        ys,
	})
}
