package main

import (
	"github.com/gomodule/redigo/redis"
	"time"
)

var pool *redis.Pool

func IntiPool(addr string, maxIdle int, maxActive int, idleTimeout time.Duration) {
	pool = &redis.Pool{
		MaxIdle:     maxIdle,     // max freedom conn number
		MaxActive:   maxActive,   // max connecting conn number, 0 standing for no limiting
		IdleTimeout: idleTimeout, // freedom times limit
		Dial: func() (redis.Conn, error) {
			return redis.Dial("tcp", addr)
		},
	}
}
