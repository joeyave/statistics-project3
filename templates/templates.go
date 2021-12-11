package templates

type StatCharacteristic struct {
	Name string
	Val  float64
	From float64
	To   float64
}

type Param struct {
	Name string
	Val  float64
}

type Stat struct {
	Name   string
	Val    float64
	P      float64
	IsNorm bool
	Alpha  float64
}

type StatWithU struct {
	Name   string
	UAbs   float64
	U      float64
	IsNorm bool
	Alpha  float64
}

type Variant struct {
	X float64
	N int
	P float64
	F float64
}

type Class struct {
	XFrom float64
	XTo   float64
	N     int
	P     float64
	F     float64
}
