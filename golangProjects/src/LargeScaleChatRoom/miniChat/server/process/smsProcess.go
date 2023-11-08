package process

import (
	"encoding/json"
	"fmt"
	"net"
	"redisLearn/miniChat/common/Message"
	"redisLearn/miniChat/common/utils"
)

type SmsProcess struct {
}

// 转发消息

func (p *SmsProcess) SendGroupMessage(mes *Message.Message) {
	// 序列化mes
	data, err := json.Marshal(mes)
	if err != nil {
		fmt.Println("Marshal SendGroupMessage Message error")
	}
	// 遍历OnlineUser，将消息转发出去
	for _, User := range userMgr.onlineUsers {
		p.SendMesToEachOnlineUser(data, User.Conn)
	}
}

func (p *SmsProcess) SendMesToEachOnlineUser(content []byte, conn net.Conn) {
	transfer := &utils.Transfer{
		Conn: conn,
	}
	transfer.WritePackage(content)
}
