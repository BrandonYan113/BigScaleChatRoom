package main

import (
	"fmt"
	"redisLearn/redigo/redis"
)

var pool *redis.Pool = &redis.Pool{
	MaxIdle:     8,   //最大空闲连接数
	MaxActive:   0,   //和数据库的最大连接数，0不限制（系统限制）
	IdleTimeout: 100, //最大空闲时间
	Dial: func() (redis.Conn, error) { //初始化连接池
		return redis.Dial("tcp", "localhost:6379")
	},
}

func main() {
	//ConnectSetup, err := redis.Dial("tcp", "localhost:6379")
	connect := pool.Get() //获取一个连接
	// pool.Close() 关闭后，无法取出连接
	//if err != nil {
	//	fmt.Println("ConnectSetup failed", err)
	//}
	fmt.Println("ConnectSetup redis successfully!")
	defer connect.Close()
	_, err := connect.Do("set", "lab", "516-1")
	if err != nil {
		fmt.Println("write error: ", err)
		return
	}
	fmt.Println("write successfully")
	data, err := redis.String(connect.Do("get", "lab"))
	fmt.Println(data)

	_, err = connect.Do("HMset", "Hero", "name", "孙悟空", "age", 500, "skill", "72变")
	if err != nil {
		fmt.Println("write stduent err: ", err)
		return
	}
	Hero, err := redis.Strings(connect.Do("HMget", "Hero", "name", "age", "skill"))
	if err != nil {
		fmt.Println("read err: ", err)
		return
	}
	fmt.Println(Hero)
	for _, cont := range Hero {
		fmt.Println(cont)
	}
}
