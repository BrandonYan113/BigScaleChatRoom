package server

import (
	"fmt"
	"io"
	"net"
	"redisLearn/miniChat/common/Message"
	"redisLearn/miniChat/common/utils"
	process2 "redisLearn/miniChat/server/process"
)

type Processor struct {
	Conn net.Conn
}

func (prc *Processor) Process() (err error) {
	//waiting client's message
	for {
		fmt.Println("waiting message from client")
		tf := &utils.Transfer{ // 建立一个Transfer来读写消息
			Conn: prc.Conn,
		}
		mes, err := tf.ReadPackage()
		if err != nil {
			if err == io.EOF {
				fmt.Println("client close server")
			} else {
				fmt.Println("read pkgError: ", err)
			}
			return err
		}
		fmt.Println("client input: ", mes)
		err = prc.ServerProcessMes(mes)
		if err != nil {
			return err
		}
	}
}

func (prc *Processor) ServerProcessMes(mes Message.Message) (err error) {
	switch mes.Type {
	case Message.LoginMesType:
		userPrc := &process2.UserProcess{
			Conn: prc.Conn,
		}
		return userPrc.ServerProcessLogin(mes)
	case Message.RegisterMesType:
		userPrc := &process2.UserProcess{
			Conn: prc.Conn,
		}
		return userPrc.ServerProcessRegister(mes)
	case Message.SmsMessageType:
		smsPrc := &process2.SmsProcess{}
		smsPrc.SendGroupMessage(&mes)
	default:
		fmt.Println("undefined message type")
	}
	return
}
