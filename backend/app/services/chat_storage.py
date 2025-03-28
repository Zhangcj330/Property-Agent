from typing import List, Optional, Dict
from datetime import datetime
from google.cloud import firestore
from google.oauth2 import service_account
from ..models import ChatMessage, ChatSession
from ..config import settings

class ChatStorageService:
    def __init__(self):
        # 使用 Firebase 配置创建凭据
        credentials = service_account.Credentials.from_service_account_info(
            settings.FIREBASE_CONFIG
        )
        
        # 使用凭据初始化 Firestore 客户端
        self.db = firestore.Client(
            credentials=credentials,
            project=settings.FIREBASE_CONFIG.get('project_id')
        )
        self.sessions_collection = self.db.collection('chat_sessions')

    async def create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """创建新的聊天会话
        
        Args:
            session_id: 可选的会话 ID。如果不提供，将自动生成
        """
        # 如果提供了 session_id，使用它；否则创建新的
        session_ref = (self.sessions_collection.document(session_id) 
                      if session_id 
                      else self.sessions_collection.document())
        
        session = ChatSession(
            session_id=session_ref.id,
            messages=[],
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        
        # 保存会话到 Firestore
        session_ref.set(session.model_dump())
        return session

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """获取会话信息"""
        try:
            session_ref = self.sessions_collection.document(session_id)
            session_doc = session_ref.get()
            
            if not session_doc.exists:
                return None
                
            session_data = session_doc.to_dict()
            
            # 转换消息列表中的时间戳
            if "messages" in session_data:
                for msg in session_data["messages"]:
                    if isinstance(msg.get("timestamp"), (int, float)):
                        msg["timestamp"] = datetime.fromtimestamp(msg["timestamp"])
            
            # 转换会话级别的时间戳
            for field in ["created_at", "last_active"]:
                if isinstance(session_data.get(field), (int, float)):
                    session_data[field] = datetime.fromtimestamp(session_data[field])
            
            return ChatSession(**session_data)
            
        except Exception as e:
            print(f"Error in get_session: {str(e)}")
            return None

    async def save_message(self, session_id: str, message: ChatMessage):
        """保存新消息到会话"""
        try:
            session_ref = self.sessions_collection.document(session_id)
            
            # 更新会话
            session_ref.update({
                "messages": firestore.ArrayUnion([message.model_dump()]),
                "last_active": datetime.now()
            })
            
        except Exception as e:
            print(f"Error in save_message: {str(e)}")
            raise

    async def update_session_state(self, session_id: str, preferences: Optional[Dict] = None, search_params: Optional[Dict] = None):
        """更新会话状态（偏好和搜索参数）"""
        try:
            session_ref = self.sessions_collection.document(session_id)
            update_data = {"last_active": datetime.now()}
            
            if preferences is not None:
                update_data["preferences"] = preferences
            if search_params is not None:
                update_data["search_params"] = search_params
                
            session_ref.update(update_data)
            
        except Exception as e:
            print(f"Error in update_session_state: {str(e)}")
            raise

    async def clear_session(self, session_id: str):
        """删除会话"""
        try:
            self.sessions_collection.document(session_id).delete()
        except Exception as e:
            print(f"Error in clear_session: {str(e)}")
            raise 