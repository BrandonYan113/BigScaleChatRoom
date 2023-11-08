package process

import (
	"encoding/json"
	"fmt"
	"redisLearn/miniChat/client/model"
	"redisLearn/miniChat/common/Message"
)

// 客户端维护的在线用户列表

var onlineUsers map[int]*Message.User = make(map[int]*Message.User, 10)
var CurUser model.CurUser // 用户登录完成后，完成CurUser的初始化

// 处理返回的NotifyUserStatusMes

func updateUserStatus(notifyUserStatusMes *Message.NotifyUserStatusMes) {
	user, ok := onlineUsers[notifyUserStatusMes.UserId]
	if ok {
		user.UserStatus = notifyUserStatusMes.Status
	} else {
		onlineUsers[notifyUserStatusMes.UserId] = &Message.User{
			UserId:     notifyUserStatusMes.UserId,
			UserStatus: notifyUserStatusMes.Status,
		}
	}
	outputUserStatus()
	return
}

// 显示online用户
func outputUserStatus() {
	for id, user := range onlineUsers {
		fmt.Printf("online users: %v %v ", id, user.UserName)
	}
}

func outputSmsMessage(mes *Message.Message) (err error) {
	var smsMes Message.SmsMes
	err = json.Unmarshal([]byte(mes.Data), &smsMes)
	if err != nil {
		fmt.Println("Unmarshal outputSmsMessage mes.data error")
		return
	}
	fmt.Printf("user %v say: %s\n\n", smsMes.User.UserId, smsMes.Content)
	return err
}
