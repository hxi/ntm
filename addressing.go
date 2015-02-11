package ntm

import (
	"log"
	"math"
)

type Similarity struct {
	U   []Unit
	V   []Unit
	Top Unit

	UV    float64
	Unorm float64
	Vnorm float64
}

func NewSimilarity(u, v []Unit) *Similarity {
	s := Similarity{
		U: u,
		V: v,
	}
	for i := 0; i < len(u); i++ {
		s.UV += u[i].Val * v[i].Val
		s.Unorm += u[i].Val * u[i].Val
		s.Vnorm += v[i].Val * v[i].Val
	}
	s.Unorm = math.Sqrt(s.Unorm)
	s.Vnorm = math.Sqrt(s.Vnorm)
	s.Top.Val = s.UV / (s.Unorm * s.Vnorm)
	if math.IsNaN(s.Top.Val) {
		log.Printf("u: %+v, v: %+v", u, v)
		panic("")
	}
	return &s
}

func (s *Similarity) Backward() {
	uvuu := s.UV / (s.Unorm * s.Unorm)
	uvvv := s.UV / (s.Vnorm * s.Vnorm)
	uvg := s.Top.Grad / (s.Unorm * s.Vnorm)
	for i, u := range s.U {
		v := s.V[i].Val
		s.U[i].Grad += (v - u.Val*uvuu) * uvg
		s.V[i].Grad += (u.Val - v*uvvv) * uvg
	}
}

type BetaSimilarity struct {
	Beta *Unit // Beta is assumed to be in the range (-Inf, Inf)
	S    *Similarity
	Top  Unit

	b float64
}

func NewBetaSimilarity(beta *Unit, s *Similarity) *BetaSimilarity {
	bs := BetaSimilarity{
		Beta: beta,
		S:    s,
		b:    beta.Val * beta.Val,
	}
	bs.Top.Val = bs.b * s.Top.Val
	return &bs
}

func (bs *BetaSimilarity) Backward() {
	bs.Beta.Grad += bs.S.Top.Val * 2 * bs.Beta.Val * bs.Top.Grad
	bs.S.Top.Grad += bs.b * bs.Top.Grad
}

type ContentAddressing struct {
	Units []*BetaSimilarity
	Top   []Unit
}

func NewContentAddressing(units []*BetaSimilarity) *ContentAddressing {
	s := ContentAddressing{
		Units: units,
		Top:   make([]Unit, len(units)),
	}
	// Increase numerical stability by subtracting all weights by their max,
	// before computing math.Exp().
	var max float64 = -1
	for _, u := range s.Units {
		max = math.Max(max, u.Top.Val)
	}
	var sum float64 = 0
	for i := 0; i < len(s.Top); i++ {
		s.Top[i].Val = math.Exp(s.Units[i].Top.Val - max)
		sum += s.Top[i].Val
	}
	for i := 0; i < len(s.Top); i++ {
		s.Top[i].Val = s.Top[i].Val / sum
	}
	return &s
}

func (s *ContentAddressing) Backward() {
	var gv float64 = 0
	for _, top := range s.Top {
		gv += top.Grad * top.Val
	}
	for i, top := range s.Top {
		s.Units[i].Top.Grad += (top.Grad - gv) * top.Val
	}
}

type GatedWeighting struct {
	G    *Unit
	WC   *ContentAddressing
	Wtm1 *Refocus // the weights at time t-1
	Top  []Unit
}

func NewGatedWeighting(g *Unit, wc *ContentAddressing, wtm1 *Refocus) *GatedWeighting {
	wg := GatedWeighting{
		G:    g,
		WC:   wc,
		Wtm1: wtm1,
		Top:  make([]Unit, len(wc.Top)),
	}
	gt := Sigmoid(g.Val)
	for i := 0; i < len(wg.Top); i++ {
		wg.Top[i].Val = gt*wc.Top[i].Val + (1-gt)*wtm1.Top[i].Val
	}
	return &wg
}

func (wg *GatedWeighting) Backward() {
	gt := Sigmoid(wg.G.Val)

	var grad float64 = 0
	for i := 0; i < len(wg.Top); i++ {
		grad += (wg.WC.Top[i].Val - wg.Wtm1.Top[i].Val) * wg.Top[i].Grad
	}
	wg.G.Grad += grad * gt * (1 - gt)

	for i := 0; i < len(wg.WC.Top); i++ {
		wg.WC.Top[i].Grad += gt * wg.Top[i].Grad
	}

	for i := 0; i < len(wg.Wtm1.Top); i++ {
		wg.Wtm1.Top[i].Grad += (1 - gt) * wg.Top[i].Grad
	}
}

type ShiftedWeighting struct {
	S   *Unit
	Z   float64
	WG  *GatedWeighting
	Top []Unit
}

func NewShiftedWeighting(s *Unit, wg *GatedWeighting) *ShiftedWeighting {
	sw := ShiftedWeighting{
		S:   s,
		WG:  wg,
		Top: make([]Unit, len(wg.Top)),
	}

	n := len(sw.WG.Top)
	sw.Z = math.Mod(math.Mod(s.Val, float64(n))+float64(n), float64(n))
	//sw.Z = float64(n) * Sigmoid(s.Val)

	simj := 1 - (sw.Z - math.Floor(sw.Z))
	for i := 0; i < len(sw.Top); i++ {
		imj := (i + int(sw.Z)) % n
		sw.Top[i].Val = sw.WG.Top[imj].Val*simj + sw.WG.Top[(imj+1)%n].Val*(1-simj)
	}
	return &sw
}

func (sw *ShiftedWeighting) Backward() {
	var grad float64 = 0
	n := len(sw.WG.Top)
	for i := 0; i < len(sw.Top); i++ {
		imj := (i + int(sw.Z)) % n
		grad += (-sw.WG.Top[imj].Val + sw.WG.Top[(imj+1)%n].Val) * sw.Top[i].Grad
	}
	//grad = grad * sw.Z * (1 - sw.Z/float64(n))
	sw.S.Grad += grad

	simj := 1 - (sw.Z - math.Floor(sw.Z))
	for i := 0; i < len(sw.WG.Top); i++ {
		j := (i - int(sw.Z) + n) % n
		sw.WG.Top[i].Grad += sw.Top[j].Grad*simj + sw.Top[(j-1+n)%n].Grad*(1-simj)
	}
}

type Refocus struct {
	Gamma *Unit
	SW    *ShiftedWeighting
	Top   []Unit

	g     float64
	maxSW float64
}

func NewRefocus(gamma *Unit, sw *ShiftedWeighting) *Refocus {
	rf := Refocus{
		Gamma: gamma,
		SW:    sw,
		Top:   make([]Unit, len(sw.Top)),
		g:     gamma.Val*gamma.Val + 1,
		maxSW: -1,
	}

	// To increase numerical stability, we divide all weights by the maximum weight,
	// so that at least the math.Pow(maxWeight, rf.g) will not be zero.
	for _, u := range sw.Top {
		rf.maxSW = math.Max(rf.maxSW, u.Val)
	}
	var sum float64 = 0
	for i := 0; i < len(rf.Top); i++ {
		rf.Top[i].Val = math.Pow(sw.Top[i].Val/rf.maxSW, rf.g)
		sum += rf.Top[i].Val
	}
	for i := 0; i < len(rf.Top); i++ {
		rf.Top[i].Val = rf.Top[i].Val / sum
	}
	return &rf
}

func (rf *Refocus) Backward() {
	var gv float64 = 0
	for _, top := range rf.Top {
		gv += top.Grad * top.Val
	}
	for i, sw := range rf.SW.Top {
		if sw.Val < machineEpsilon {
			continue
		}
		top := rf.Top[i]
		rf.SW.Top[i].Grad += (top.Grad - gv) * rf.g / sw.Val * top.Val
	}

	lns := make([]float64, len(rf.SW.Top))
	var lnexp float64 = 0
	var s float64 = 0
	for i, sw := range rf.SW.Top {
		if sw.Val < machineEpsilon {
			continue
		}
		lns[i] = math.Log(sw.Val)
		pow := math.Pow(sw.Val/rf.maxSW, rf.g)
		lnexp += lns[i] * pow
		s += pow
	}
	lnexps := lnexp / s
	var grad float64 = 0
	for i, top := range rf.Top {
		if rf.SW.Top[i].Val < machineEpsilon {
			continue
		}
		grad += top.Grad * (top.Val * (lns[i] - lnexps))
	}
	grad = grad * 2 * rf.Gamma.Val
	rf.Gamma.Grad += grad
}

type Read struct {
	W      *Refocus
	Memory *WrittenMemory
	Top    []Unit
}

func NewRead(w *Refocus, memory *WrittenMemory) *Read {
	r := Read{
		W:      w,
		Memory: memory,
		Top:    make([]Unit, len(memory.Top[0])),
	}
	for i := 0; i < len(r.Top); i++ {
		var v float64 = 0
		for j := 0; j < len(w.Top); j++ {
			v += w.Top[j].Val * memory.Top[j][i].Val
		}
		r.Top[i].Val = v
	}
	return &r
}

func (r *Read) Backward() {
	for i := 0; i < len(r.W.Top); i++ {
		var grad float64 = 0
		for j := 0; j < len(r.Top); j++ {
			grad += r.Top[j].Grad * r.Memory.Top[i][j].Val
		}
		r.W.Top[i].Grad += grad
	}

	for i := 0; i < len(r.Memory.Top); i++ {
		for j := 0; j < len(r.Memory.Top[i]); j++ {
			r.Memory.Top[i][j].Grad += r.Top[j].Grad * r.W.Top[i].Val
		}
	}
}

type WrittenMemory struct {
	Ws    []*Refocus
	Heads []*Head        // We actually need only the erase and add vectors.
	Mtm1  *WrittenMemory // memory at time t-1
	Top   [][]Unit

	erase    [][]float64
	add      [][]float64
	erasures [][]float64
}

func NewWrittenMemory(ws []*Refocus, heads []*Head, mtm1 *WrittenMemory) *WrittenMemory {
	wm := WrittenMemory{
		Ws:    ws,
		Heads: heads,
		Mtm1:  mtm1,
		Top:   makeTensorUnit2(len(mtm1.Top), len(mtm1.Top[0])),

		erase:    MakeTensor2(len(heads), len(mtm1.Top[0])),
		add:      MakeTensor2(len(heads), len(mtm1.Top[0])),
		erasures: MakeTensor2(len(mtm1.Top), len(mtm1.Top[0])),
	}
	for i, h := range wm.Heads {
		erase := wm.erase[i]
		add := wm.add[i]
		eraseVec := h.EraseVector()
		addVec := h.AddVector()
		for j, e := range eraseVec {
			erase[j] = Sigmoid(e.Val)
			add[j] = Sigmoid(addVec[j].Val)
		}
	}
	for i, mtm1Row := range wm.Mtm1.Top {
		erasure := wm.erasures[i]
		topRow := wm.Top[i]
		for j, mtm1 := range mtm1Row {
			var e float64 = 1
			var adds float64 = 0
			for k, weights := range wm.Ws {
				e = e * (1 - weights.Top[i].Val*wm.erase[k][j])
				adds += weights.Top[i].Val * wm.add[k][j]
			}
			erasure[j] = e
			topRow[j].Val += erasure[j]*mtm1.Val + adds
		}
	}
	return &wm
}

func (wm *WrittenMemory) Backward() {
	// Gradient of wtm1, erase and add vectors
	var grad float64 = 0
	for i, weights := range wm.Ws {
		hErase := wm.Heads[i].EraseVector()
		hAdd := wm.Heads[i].AddVector()
		erase := wm.erase[i]
		add := wm.add[i]
		for j, topRow := range wm.Top {
			wtm1 := weights.Top[j].Val
			mtm1Row := wm.Mtm1.Top[j]
			grad = 0
			for k, top := range topRow {
				mtilt := mtm1Row[k].Val
				for q, ws := range wm.Ws {
					if q == i {
						continue
					}
					mtilt = mtilt * (1 - ws.Top[j].Val*wm.erase[q][k])
				}
				grad += (mtilt*(-erase[k]) + add[k]) * top.Grad
				hErase[k].Grad += mtilt * (-wtm1) * top.Grad
				hAdd[k].Grad += wtm1 * top.Grad
			}
			weights.Top[j].Grad += grad
		}
		for k, e := range erase {
			hErase[k].Grad = hErase[k].Grad * e * (1 - e)
		}
		for k, a := range add {
			hAdd[k].Grad = hAdd[k].Grad * a * (1 - a)
		}
	}

	// Gradient of wm.Mtm1
	for i, topRow := range wm.Top {
		mtm1Row := wm.Mtm1.Top[i]
		eRow := wm.erasures[i]
		for j, top := range topRow {
			mtm1Row[j].Grad += eRow[j] * top.Grad
		}
	}
}

type Circuit struct {
	W  []*Refocus
	R  []*Read
	WM *WrittenMemory
}

func NewCircuit(heads []*Head, mtm1 *WrittenMemory) *Circuit {
	circuit := Circuit{
		R: make([]*Read, len(heads)),
	}
	circuit.W = make([]*Refocus, len(heads))
	for i, h := range heads {
		ss := make([]*BetaSimilarity, len(mtm1.Top))
		for j := range mtm1.Top {
			s := NewSimilarity(h.K(), mtm1.Top[j])
			ss[j] = NewBetaSimilarity(h.Beta(), s)
		}
		wc := NewContentAddressing(ss)
		wg := NewGatedWeighting(h.G(), wc, h.Wtm1)
		ws := NewShiftedWeighting(h.S(), wg)
		circuit.W[i] = NewRefocus(h.Gamma(), ws)
		circuit.R[i] = NewRead(circuit.W[i], mtm1)
	}

	circuit.WM = NewWrittenMemory(circuit.W, heads, mtm1)
	return &circuit
}

func (c *Circuit) Backward() {
	for _, r := range c.R {
		r.Backward()
	}
	c.WM.Backward()

	for _, rf := range c.WM.Ws {
		rf.Backward()
		rf.SW.Backward()
		rf.SW.WG.Backward()
		rf.SW.WG.WC.Backward()
		for _, bs := range rf.SW.WG.WC.Units {
			bs.Backward()
			bs.S.Backward()
		}
	}
}

func (c *Circuit) ReadVals() [][]float64 {
	res := MakeTensor2(len(c.R), len(c.R[0].Top))
	for i := 0; i < len(res); i++ {
		for j := 0; j < len(res[i]); j++ {
			res[i][j] = c.R[i].Top[j].Val
		}
	}
	return res
}

func (c *Circuit) WrittenMemoryVals() [][]float64 {
	res := MakeTensor2(len(c.WM.Top), len(c.WM.Top[0]))
	for i := 0; i < len(res); i++ {
		for j := 0; j < len(res[i]); j++ {
			res[i][j] = c.WM.Top[i][j].Val
		}
	}
	return res
}
