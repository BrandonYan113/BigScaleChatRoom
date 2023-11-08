package process

import (
	"encoding/json"
	"fmt"
	"net"
	"redisLearn/miniChat/common/Message"
	"redisLearn/miniChat/common/utils"
	"redisLearn/miniChat/server/model"
)

type UserProcess struct {
	Conn   net.Conn
	UserId int
}

func (UP *UserProcess) ServerProcessLogin(mes Message.Message) (err error) {
	var Mes Message.LoginMes
	err = json.Unmarshal([]byte(mes.Data), &Mes)
	if err != nil {
		fmt.Println("unmarshal Login Message err: ", err)
		return
	}

	var loginResp Message.Message
	loginResp.Type = Message.LoginReturnMesType

	var RespMes Message.LoginReturnMes

	// check
	user, err := model.GlobaluserDo.HandleLogin(Mes.ID, Mes.Password)
	if err == nil {
		RespMes.Code = 200
		// 添加到在线列表
		UP.UserId = user.ID
		userMgr.AddOnlineUsers(UP)
		// 通知他人自己已经上线
		UP.NotifyOtherUsersOnlineUser(user.ID)
		// RespMes
		for id, _ := range userMgr.onlineUsers {
			RespMes.Users = append(RespMes.Users, id)
		}
		fmt.Println(user, " model successful, online list: ", RespMes.Users)
	} else { // illegal
		if err == model.ERROR_USER_NOT_EXITST {
			RespMes.Code = 500
			RespMes.Err = err.Error()
		} else if err == model.ERROR_PASSWORD_NOT_CORRECT {
			RespMes.Code = 403
			RespMes.Err = err.Error()
		} else {
			RespMes.Code = 505
			RespMes.Err = "Not define error"
		}
	}
	data, err := json.Marshal(RespMes)
	if err != nil {
		fmt.Println("model response message marshal err: ", err)
		return
	}
	loginResp.Data = string(data)
	data, err = json.Marshal(loginResp)
	if err != nil {
		fmt.Println("model response message marshal err: ", err)
	}
	tf := &utils.Transfer{
		Conn: UP.Conn,
	}
	err = tf.WritePackage(data)

	return err
}

func (UP *UserProcess) ServerProcessRegister(mes Message.Message) (err error) {
	var RegMes Message.Message
	RegMes.Type = Message.RegisterMesType

	var Mes Message.RegisterMes
	err = json.Unmarshal([]byte(mes.Data), &Mes)
	if err != nil {
		fmt.Println("unmarshal Login Message err: ", err)
		return
	}

	var RegisterResp Message.Message
	RegisterResp.Type = Message.RegisterReturnMesType

	var RespMes Message.RegisterResponseMes

	// check user exist or not
	err = model.GlobaluserDo.HandleRegister(&Mes.User)
	if err == nil {
		RespMes.Code = 200
	} else { // illegal
		if err == model.ERROR_USER_HAS_EXIST {
			RespMes.Code = 400
			RespMes.Error = err.Error()
		} else {
			RespMes.Code = 505
			RespMes.Error = "Not define error"
		}
	}
	data, err := json.Marshal(RespMes)
	if err != nil {
		fmt.Println("Register response message marshal err: ", err)
		return
	}
	RegisterResp.Data = string(data)
	data, err = json.Marshal(RegisterResp)
	if err != nil {
		fmt.Println("model response message marshal err: ", err)
	}
	tf := &utils.Transfer{
		Conn: UP.Conn,
	}
	err = tf.WritePackage(data)

	return err
}

// 通知他人我已经上线

func (p *UserProcess) NotifyOtherUsersOnlineUser(id int) {
	for userId, User := range userMgr.onlineUsers {
		if userId == p.UserId { //不通知自己
			continue
		}
		// 通知他人
		User.NotifyOthersImOnline(id)
	}
}

func (p *UserProcess) NotifyOthersImOnline(id int) {
	// 消息实体
	var mes Message.Message
	mes.Type = Message.NotifyUserStatusMesType

	// 消息结构体
	mesStrcut := Message.NotifyUserStatusMes{UserId: id, Status: Message.UserOnlineStatus}
	SerializeMesStruct, err := json.Marshal(mesStrcut)
	if err != nil {
		fmt.Println("serialize NotifyOthersImOnline Message struct error")
		return
	}
	mes.Data = string(SerializeMesStruct)
	SerialMes, err := json.Marshal(mes)
	if err != nil {
		fmt.Println("serialize NotifyOthersImOnline Message error")
		return
	}
	transfer := utils.Transfer{
		Conn: p.Conn,
	}
	err = transfer.WritePackage(SerialMes)
	if err != nil {
		fmt.Println("write NotifyOthersImOnline Message error")
		return
	}
}
