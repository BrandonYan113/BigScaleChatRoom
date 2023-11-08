package process

import (
	"encoding/json"
	"fmt"
	"redisLearn/miniChat/common/Message"
	"redisLearn/miniChat/common/utils"
)

type SmsProcess struct {
}

// 群聊消息

func (p *SmsProcess) SendGroupMessage(content string) (err error) {
	// 消息实体
	var mes Message.Message
	mes.Type = Message.SmsMessageType

	// 消息结构体
	var smsStruct Message.SmsMes
	smsStruct.Content = content
	smsStruct.User = &CurUser.User

	data, err := json.Marshal(smsStruct)
	if err != nil {
		fmt.Println("Marshal 'SendGroupMessage' smsStruct error")
		return err
	}
	mes.Data = string(data)
	data, err = json.Marshal(mes)
	if err != nil {
		fmt.Println("Marshal 'SendGroupMessage' mes error")
		return err
	}
	trans := &utils.Transfer{
		Conn: CurUser.Conn,
	}
	err = trans.WritePackage(data)
	if err != nil {
		fmt.Println("send 'SendGroupMessage' message error")
	}
	return err
}
