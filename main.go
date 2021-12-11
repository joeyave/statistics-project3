package main

import (
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/joeyave/statistics-project3/controllers"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"html/template"
	"os"
	"reflect"
	"strings"
	"time"
)

func main() {

	out := zerolog.ConsoleWriter{
		Out:        os.Stdout,
		TimeFormat: time.RFC3339,
	}
	log.Logger = zerolog.New(out).Level(zerolog.GlobalLevel()).With().Timestamp().Logger()

	router := gin.New()
	router.SetFuncMap(template.FuncMap{
		"sub": func(i, j int) int {
			return i - j
		},
		"add": func(i, j int) int {
			return i + j
		},
		"avail": func(name string, data interface{}) bool {
			v := reflect.ValueOf(data)
			if v.Kind() == reflect.Ptr {
				v = v.Elem()
			}
			if v.Kind() != reflect.Struct {
				return false
			}
			return v.FieldByName(name).IsValid()
		},
		"printf": func(f string, x interface{}) string {
			s := fmt.Sprintf(f, x)
			return strings.TrimRight(strings.TrimRight(s, "0"), ".")
		},
	})

	router.Static("/css", "./css")
	router.LoadHTMLGlob("templates/*")

	router.GET("/index", controllers.Index)
	router.POST("/upload", controllers.Upload)
	router.Any("/identifyNormDistribution", controllers.IdentifyNormDistribution)

	log.Info().Msgf("Starting Gin with mode: %s", gin.Mode())
	err := router.Run()
	if err != nil {
		log.Fatal().Msgf("Error starting Gin: %v", err)
	}
}
