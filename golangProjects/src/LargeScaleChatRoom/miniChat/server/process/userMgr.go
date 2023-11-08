package process

import "fmt"

var (
	userMgr *UserMgr
)

type UserMgr struct {
	onlineUsers map[int]*UserProcess `json:"onlineUsers"`
}

func init() {
	userMgr = &UserMgr{
		onlineUsers: make(map[int]*UserProcess, 1024),
	}
}

func (p *UserMgr) AddOnlineUsers(up *UserProcess) {
	p.onlineUsers[up.UserId] = up

}

func (p *UserMgr) DelOnlineUsers(userID int) {
	delete(p.onlineUsers, userID)
}

func (p *UserMgr) GetAllOnlineUsers() map[int]*UserProcess {
	return p.onlineUsers
}

// 根据id返回对应值

func (p *UserMgr) GetOnlineUserById(userId int) (user *UserProcess, err error) {
	user, ok := p.onlineUsers[userId]
	if !ok {
		err = fmt.Errorf("id does not exist")
	}
	return
}
