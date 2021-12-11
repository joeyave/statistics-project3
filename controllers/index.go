package controllers

import (
	"github.com/gin-gonic/gin"
	"github.com/joeyave/statistics-project3/global"
	"net/http"
)

func Index(c *gin.Context) {
	c.HTML(http.StatusOK, "index.tmpl", map[string]interface{}{
		"FileName": global.FileName(),
	})
}
