package process

import (
	"encoding/json"
	"fmt"
	"net"
	"redisLearn/miniChat/common/ConnectSetup"
	"redisLearn/miniChat/common/Message"
	"redisLearn/miniChat/common/utils"
	"redisLearn/miniChat/server/model"
)

type UserProcess struct {
}

func (prc *UserProcess) Login(id int, psw string) (err error) {
	conn, _ := net.Dial("tcp", ConnectSetup.Localhost)
	defer conn.Close()
	var mes Message.Message
	var LoginMes Message.LoginMes

	mes.Type = Message.LoginMesType
	LoginMes.ID = id
	LoginMes.Password = psw
	mesStr, err := json.Marshal(LoginMes)
	if err != nil {
		fmt.Println("serialization LoginMes failed: ", err)
		return
	}
	mes.Data = string(mesStr)
	data, err := json.Marshal(mes)
	if err != nil {
		fmt.Println("mes serialization error: ", err)
		return
	}
	// sent message
	tf := &utils.Transfer{
		Conn: conn,
	}
	err = tf.WritePackage(data)
	if err != nil {
		fmt.Println("sent data err: ", err)
	}

	// handle with response message
	response, _ := tf.ReadPackage()
	var LoginRespMes Message.LoginReturnMes
	err = json.Unmarshal([]byte(response.Data), &LoginRespMes)
	if LoginRespMes.Code == 200 {
		// 维护在线用户的连接
		CurUser.UserId = id
		CurUser.Conn = conn
		CurUser.UserStatus = Message.UserOnlineStatus
		// keep a goroutine connecting with server
		// 当前在线用户列表
		fmt.Println("current online users list:  ")
		for _, cid := range LoginRespMes.Users {
			if cid == id {
				continue
			}
			fmt.Printf("%v \n", cid)
			onlineUsers[cid] = &Message.User{
				UserId:     cid,
				UserStatus: Message.UserOnlineStatus,
			}
		}
		fmt.Println()
		go processServerMes(conn)

		// show Login successful menu
		for {
			ShowMenu(CurUser.UserName)
		}
	} else {
		fmt.Println("model failed: ", err, "LoginRespMes.Code: ", LoginRespMes.Code)
	}
	return
}

func processServerMes(conn net.Conn) {
	trans := &utils.Transfer{
		Conn: conn,
	}
	for {
		//fmt.Println("server is waiting message from client: ", &conn)
		Mes, err := trans.ReadPackage()
		if err != nil {
			fmt.Println("client read package err: ", err)
			return
		}
		//fmt.Println(Mes)
		switch Mes.Type {
		case Message.NotifyUserStatusMesType: // 通知服务器有人上线了
			// 处理消息，显示上线的用户id和状态, 保存
			var notifyUserStatusMes Message.NotifyUserStatusMes
			json.Unmarshal([]byte(Mes.Data), &notifyUserStatusMes)
			updateUserStatus(&notifyUserStatusMes)
		case Message.SmsMessageType: // 打印群消息
			fmt.Println("group message")
			outputSmsMessage(&Mes)
		default:
			fmt.Printf("undefined message Type: %v\n", Mes.Type)
		}
	}
}

func (p *UserProcess) Register(userName string, id int, psw string) (err error) {
	conn, _ := net.Dial("tcp", ConnectSetup.Localhost)
	defer conn.Close()
	var mes Message.Message
	var RegisterMes Message.RegisterMes
	var User model.User

	mes.Type = Message.RegisterMesType
	User.ID = id
	User.UserName = userName
	User.Password = psw
	RegisterMes.User = User

	mesStr, err := json.Marshal(RegisterMes)
	if err != nil {
		fmt.Println("serialization RegisterMes failed: ", err)
		return
	}
	mes.Data = string(mesStr)
	data, err := json.Marshal(mes)
	if err != nil {
		fmt.Println("mes serialization error: ", err)
		return
	}

	// sent message of register
	tf := &utils.Transfer{
		Conn: conn,
	}
	fmt.Println(mes)
	err = tf.WritePackage(data)
	if err != nil {
		fmt.Println("sent data err: ", err)
	}

	// waiting server response for register
	// handle with response message from sever
	response, _ := tf.ReadPackage()
	var RegisterResponMes Message.RegisterResponseMes
	//fmt.Println(response.Data)
	err = json.Unmarshal([]byte(response.Data), &RegisterResponMes)
	if RegisterResponMes.Code == 200 {
		// keep a goroutine connecting with server
		go processServerMes(conn)

		// show Register successful menu
		for {
			ShowMenu(userName)
		}
	} else {
		fmt.Println("Register failed: ", err, "RegisterResponMes.Code: ", RegisterResponMes.Code)
	}
	return
}
