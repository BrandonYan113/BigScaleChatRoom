package main

import (
	"fmt"
	"net"
	"redisLearn/miniChat/common/ConnectSetup"
	"redisLearn/miniChat/server"
	"redisLearn/miniChat/server/model"
	"time"
)

func init() {
	IntiPool("localhost:6379", 16, 0, 300*time.Second)
	initUserDo()
}

func mainProcess(conn net.Conn) {
	defer conn.Close()
	processor := &server.Processor{
		Conn: conn,
	}
	err := processor.Process()
	if err != nil {
		fmt.Println("mainProcess error, exit: ", err)
	}
}

func initUserDo() {
	model.GlobaluserDo = model.NewUserDo(pool) // this pool come from redis.go
}

func main() {
	fmt.Println("server surveillance at port 8890")
	listen, err := net.Listen("tcp", ConnectSetup.Localhost)
	if err != nil {
		fmt.Println("listen err: ", err)
		return
	}
	defer listen.Close()
	for {
		conn, err := listen.Accept()
		if err != nil {
			fmt.Println("listen accept err: ", err)
		}
		go mainProcess(conn)
	}

}
