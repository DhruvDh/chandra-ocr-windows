// New code, MPL-2.0. Source reference Transformers 5.18 Qwen3.5.
#include "text_model.h"
#include <stdexcept>
#include <algorithm>
namespace chandra::dc {
namespace {
struct P { uint32_t rows,width,offset,stride,base,count,mode,pad; };
Buffer slice(Device& d,const Buffer& x,uint32_t rows,uint32_t width,uint32_t offset,uint32_t stride) {
 Buffer y=d.floats(rows*width);P p{rows,width,offset,stride,0,0,0,0};d.dispatch("text_slice",{&x},{&y},&p,sizeof(p),(rows*width+127)/128);return y;
}
void positions(const TextPositions& p,uint32_t n){for(auto axis:{&p.temporal,&p.height,&p.width})for(auto v:*axis)if(v>=262144)throw std::runtime_error("MRoPE position outside pinned maximum");if(p.temporal.size()!=n||p.height.size()!=n||p.width.size()!=n)throw std::runtime_error("Three explicit MRoPE position axes required");}
const std::string root="model.language_model.";
}
TextModel::TextModel(Device& d,ModelWeights& w):device(d),weights(w){}
TextRequest TextModel::newRequest(){TextRequest r;for(uint32_t l=0;l<32;l++){auto& c=r.layers[l];if(l%4==3){c.keys=device.floats(16384*1024);c.values=device.floats(16384*1024);}else{c.conv=device.floats(8192*4);c.recurrent=device.floats(32*128*128);}device.zero(l%4==3?c.keys:c.conv);device.zero(l%4==3?c.values:c.recurrent);}return r;}
void TextModel::retire(TextRequest& r) noexcept {r.failed=true;r.retired=true;r.layers={};}
Buffer TextModel::mix(uint32_t l,const Buffer& x,uint32_t n,const TextPositions& pos,TextLayerCache& c){
 std::string b=root+"layers."+std::to_string(l)+".";
 if(c.failed)throw std::runtime_error("Poisoned layer cache must be retired");
 if(c.length>16384||n>16384-c.length)throw std::runtime_error("Context capacity exceeded");
 if(l%4==3){
  b+="self_attn.";auto qg=linear(device,x,weights.at(b+"q_proj.weight"),n);auto q=slice(device,qg,n*16,256,0,512);auto gate=slice(device,qg,n*16,256,256,512);
  auto k=linear(device,x,weights.at(b+"k_proj.weight"),n);auto v=linear(device,x,weights.at(b+"v_proj.weight"),n);
  q=rmsNorm(device,q,weights.at(b+"q_norm.weight"),n*16,256,1e-6f,true);k=rmsNorm(device,k,weights.at(b+"k_norm.weight"),n*4,256,1e-6f,true);
  auto pt=device.words(n,pos.temporal.data()),ph=device.words(n,pos.height.data()),pw=device.words(n,pos.width.data());auto qr=device.floats(n*4096),kr=device.floats(n*1024);P p{n,16,0,0,0,0,0,0};device.dispatch("text_rope",{&q,&pt,&ph,&pw},{&qr},&p,sizeof(p),(n*4096+127)/128);p.width=4;device.dispatch("text_rope",{&k,&pt,&ph,&pw},{&kr},&p,sizeof(p),(n*1024+127)/128);
  p={n,1024,0,0,c.length,0,0,0};device.dispatch("text_cache",{&kr,&v},{&c.keys,&c.values},&p,sizeof(p),(n*1024+127)/128);
  Buffer scores=device.floats(n*16*(c.length+n)),out=device.floats(n*4096);p={n,16,0,0,c.length,c.length+n,0,0};
  // Query groups own one query/head; bounded 128-key tiles avoid a long dot-product dispatch.
  device.dispatch("text_attention_scores",{&qr,&c.keys},{&scores},&p,sizeof(p),(p.count+127)/128,n*16);
  device.dispatch("text_attention_softmax",{},{&scores},&p,sizeof(p),n*16);
  uint32_t totalKeys=p.count,tiles=(totalKeys+127)/128;auto partials=device.floats(tiles*n*4096);for(uint32_t tile=0;tile<tiles;tile++){p.base=tile*128;p.offset=tile;device.dispatch("text_attention_values",{&scores,&c.values},{&partials},&p,sizeof(p),16,n);}p.base=0;p.count=tiles;device.dispatch("text_attention_reduce",{&partials},{&out},&p,sizeof(p),(n*4096+127)/128);
  auto sg=unary(device,gate,n*4096,2);out=binary(device,out,sg,n*4096,1);c.length+=n;return linear(device,out,weights.at(b+"o_proj.weight"),n);
 }
 b+="linear_attn.";auto qkv=linear(device,x,weights.at(b+"in_proj_qkv.weight"),n);auto z=linear(device,x,weights.at(b+"in_proj_z.weight"),n);auto a=linear(device,x,weights.at(b+"in_proj_a.weight"),n);auto beta=linear(device,x,weights.at(b+"in_proj_b.weight"),n);
 Buffer cv=device.floats(n*8192);auto& cw=weights.at(b+"conv1d.weight");if(cw.shards.size()!=1)throw std::runtime_error("Small conv weight must be one shard");P p{n,8192,0,0,0,0,0,0};device.dispatch("text_conv",{&qkv,&cw.shards[0]},{&cv,&c.conv},&p,sizeof(p),64);
 Buffer q=device.floats(n*4096),k=device.floats(n*4096),v=device.floats(n*4096),g=device.floats(n*32),bt=device.floats(n*32);
 auto& aw=weights.at(b+"A_log");auto& dt=weights.at(b+"dt_bias");if(aw.shards.size()!=1||dt.shards.size()!=1)throw std::runtime_error("Small GDN weights must be single shards");
 p={n,32,0,0,0,0,0,0};device.dispatch("text_gdn_prepare",{&cv,&a,&beta,&aw.shards[0],&dt.shards[0]},{&q,&k,&v,&g,&bt},&p,sizeof(p),32,n);
 Buffer recurrentOut=device.floats(n*4096);p={32,128,128,n,0,0,0,0};device.dispatch("text_delta",{&q,&k,&v,&g,&bt},{&c.recurrent,&recurrentOut},&p,sizeof(p),32);
 auto& nw=weights.at(b+"norm.weight");Buffer gated=device.floats(n*4096);p={n,32,128,0,0,0,0,0};device.dispatch("text_gdn_gate",{&recurrentOut,&z,&nw.shards.at(0)},{&gated},&p,sizeof(p),32,n);c.length+=n;return linear(device,gated,weights.at(b+"out_proj.weight"),n);
}
Buffer TextModel::layer(uint32_t l,const Buffer& x,uint32_t n,const TextPositions& p,TextLayerCache& c){auto b=root+"layers."+std::to_string(l)+".";auto h=rmsNorm(device,x,weights.at(b+"input_layernorm.weight"),n,2560,1e-6f,true);h=mix(l,h,n,p,c);h=binary(device,x,h,n*2560,0);auto norm=rmsNorm(device,h,weights.at(b+"post_attention_layernorm.weight"),n,2560,1e-6f,true);auto gate=linear(device,norm,weights.at(b+"mlp.gate_proj.weight"),n),up=linear(device,norm,weights.at(b+"mlp.up_proj.weight"),n);auto act=unary(device,gate,n*9216,0);auto product=binary(device,act,up,n*9216,1);auto down=linear(device,product,weights.at(b+"mlp.down_proj.weight"),n);auto result=binary(device,h,down,n*2560,0);device.drain();return result;}
Buffer TextModel::diagnosticLayer(uint32_t l,const Buffer& h,uint32_t n,const TextPositions& p,TextLayerCache& c){if(l>=32||n==0||n>64)throw std::runtime_error("Layer diagnostic bounded to64tokens");positions(p,n);if(c.failed||c.length>16384||n>16384-c.length||!h.storage||h.packedBF16||h.logicalElements!=uint64_t(n)*2560||h.words!=n*2560||(l%4==3?(!c.keys.storage||!c.values.storage||c.keys.words!=16384*1024||c.values.words!=16384*1024||c.keys.packedBF16||c.values.packedBF16):(!c.conv.storage||!c.recurrent.storage||c.conv.words!=8192*4||c.recurrent.words!=32*128*128||c.conv.packedBF16||c.recurrent.packedBF16)))throw std::runtime_error("Layer diagnostic shape/cache invalid");try{return layer(l,h,n,p,c);}catch(...){c.failed=true;throw;}}
TextResult TextModel::prefill(TextRequest& r,const Buffer& input,const TextPositions& pos,bool last){if(pos.temporal.size()>16384)throw std::runtime_error("Input token count exceeds fixed context");uint32_t n=uint32_t(pos.temporal.size());positions(pos,n);if(r.failed||r.retired)throw std::runtime_error("Failed/retired request cannot be reused");if(r.contextLimit!=16384||r.outputLimit!=12384||!n||r.tokens>16384||n>16384-r.tokens||!input.storage||input.packedBF16||input.logicalElements!=uint64_t(n)*2560||input.words!=n*2560)throw std::runtime_error("Explicit full context/embedding shape required");for(uint32_t l=0;l<32;l++){const auto& c=r.layers[l];if(c.failed||c.length!=r.tokens||(l%4==3?(!c.keys.storage||!c.values.storage||c.keys.words!=16384*1024||c.values.words!=16384*1024||c.keys.packedBF16||c.values.packedBF16):(!c.conv.storage||!c.recurrent.storage||c.conv.words!=8192*4||c.recurrent.words!=32*128*128||c.conv.packedBF16||c.recurrent.packedBF16)))throw std::runtime_error("Request cache identity/length/shape inconsistent");}try{Buffer final;for(uint32_t start=0;start<n;start+=64){uint32_t count=std::min(64u,n-start);auto h=slice(device,input,count,2560,start*2560,2560);TextPositions pp;for(auto pair:{std::pair{&pp.temporal,&pos.temporal},std::pair{&pp.height,&pos.height},std::pair{&pp.width,&pos.width}})pair.first->assign(pair.second->begin()+start,pair.second->begin()+start+count);for(uint32_t l=0;l<32;l++)h=layer(l,h,count,pp,r.layers[l]);h=rmsNorm(device,h,weights.at(root+"norm.weight"),count,2560,1e-6f,true);if(last)final=slice(device,h,1,2560,(count-1)*2560,2560);else{if(!final.storage)final=device.floats(n*2560);P p{count,2560,0,0,start,0,0,0};device.dispatch("text_copy",{&h},{&final},&p,sizeof(p),(count*2560+127)/128);}device.drain();}r.tokens+=n;auto logits=linear(device,final,weights.at(root+"embed_tokens.weight"),last?1:n,true);device.drain();return {final,logits,last?1:n};}catch(...){r.failed=true;for(auto& c:r.layers)c.failed=true;throw;}}
TextResult TextModel::prefill(TextRequest& r,const std::vector<uint32_t>& ids,const TextPositions& p,bool last){if(r.failed||r.retired)throw std::runtime_error("Failed/retired request cannot be reused");for(auto id:ids)if(id>=248320)throw std::runtime_error("Token outside pinned vocabulary");positions(p,uint32_t(ids.size()));auto h=embedding(device,weights.at(root+"embed_tokens.weight"),ids);return prefill(r,h,p,last);}
TextResult TextModel::advance(TextRequest& r,uint32_t id,const TextPositions& p){if(r.generated>=r.outputLimit||p.temporal.size()!=1)throw std::runtime_error("Retained output allowance exhausted");auto result=prefill(r,std::vector<uint32_t>{id},p,true);r.generated++;return result;}
}
