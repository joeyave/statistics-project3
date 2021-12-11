package global

var g Global

func init() {
	g = Global{}
}

type Global struct {
	x        []float64
	y        []float64
	fileName string
	dataType string
}

func CopyX() []float64 {
	var xCopy = make([]float64, len(g.x))
	copy(xCopy, g.x)
	return xCopy
}

func SetX(x []float64) {
	g.x = x
}

func CopyY() []float64 {
	var yCopy = make([]float64, len(g.y))
	copy(yCopy, g.y)
	return yCopy
}

func SetY(y []float64) {
	g.y = y
}

func FileName() string {
	return g.fileName
}

func SetFileName(fileName string) {
	g.fileName = fileName
}

func SetDataType(dataType string) {
	g.dataType = dataType
}

func DataType() string {
	return g.dataType
}
