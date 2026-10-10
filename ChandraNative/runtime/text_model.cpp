// New code, MPL-2.0. Source reference Transformers 5.18 Qwen3.5.
#include "text_model.h"
#include "padded32_opt_in.h"
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <unordered_set>
namespace chandra::dc::experimental { const char* gemvB1Selection(); }
namespace chandra::dc {
namespace {
struct P { uint32_t rows,width,offset,stride,base,count,mode,pad; };
Buffer slice(Device& d,const Buffer& x,uint32_t rows,uint32_t width,uint32_t offset,uint32_t stride) {
 Buffer y=d.floats(rows*width);P p{rows,width,offset,stride,0,0,0,0};d.dispatch("text_slice",{&x},{&y},&p,sizeof(p),(rows*width+127)/128);return y;
}
void positions(const TextPositions& p,uint32_t n){for(auto axis:{&p.temporal,&p.height,&p.width})for(auto v:*axis)if(v>=262144)throw std::runtime_error("MRoPE position outside pinned maximum");if(p.temporal.size()!=n||p.height.size()!=n||p.width.size()!=n)throw std::runtime_error("Three explicit MRoPE position axes required");}
const std::string root="model.language_model.";
// Observation only: called after layer() drained, so cache/hidden values are complete for this chunk.
void observeLayer(const TextObserver& o,uint32_t l,const Buffer& h,const TextLayerCache& c,uint32_t before,uint32_t generated,uint32_t n,uint32_t start,uint32_t count,const TextPositions& pos,const std::vector<uint32_t>* ids){
 o({TextStage::LayerOutput,l,h,count,2560,before,generated,n,start,count,c.length,pos,ids});
 if(l%4!=3){o({TextStage::ConvState,l,c.conv,8192,4,before,generated,n,start,count,c.length,pos,ids});o({TextStage::RecurrentState,l,c.recurrent,32,128*128,before,generated,n,start,count,c.length,pos,ids});}
}
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
TextResult TextModel::prefill(TextRequest& r,const Buffer& input,const TextPositions& pos,bool last,const TextObserver& observer){return forward(r,input,pos,last,observer,nullptr);}
TextResult TextModel::forward(TextRequest& r,const Buffer& input,const TextPositions& pos,bool last,const TextObserver& observer,const std::vector<uint32_t>* ids){if(pos.temporal.size()>16384)throw std::runtime_error("Input token count exceeds fixed context");uint32_t n=uint32_t(pos.temporal.size());positions(pos,n);if(r.failed||r.retired)throw std::runtime_error("Failed/retired request cannot be reused");if(r.contextLimit!=16384||r.outputLimit!=12384||!n||r.tokens>16384||n>16384-r.tokens||!input.storage||input.packedBF16||input.logicalElements!=uint64_t(n)*2560||input.words!=n*2560)throw std::runtime_error("Explicit full context/embedding shape required");for(uint32_t l=0;l<32;l++){const auto& c=r.layers[l];if(c.failed||c.length!=r.tokens||(l%4==3?(!c.keys.storage||!c.values.storage||c.keys.words!=16384*1024||c.values.words!=16384*1024||c.keys.packedBF16||c.values.packedBF16):(!c.conv.storage||!c.recurrent.storage||c.conv.words!=8192*4||c.recurrent.words!=32*128*128||c.conv.packedBF16||c.recurrent.packedBF16)))throw std::runtime_error("Request cache identity/length/shape inconsistent");}const uint32_t before=r.tokens,generated=r.generated;try{Buffer final;for(uint32_t start=0;start<n;start+=64){uint32_t count=std::min(64u,n-start);auto h=slice(device,input,count,2560,start*2560,2560);TextPositions pp;for(auto pair:{std::pair{&pp.temporal,&pos.temporal},std::pair{&pp.height,&pos.height},std::pair{&pp.width,&pos.width}})pair.first->assign(pair.second->begin()+start,pair.second->begin()+start+count);for(uint32_t l=0;l<32;l++){h=layer(l,h,count,pp,r.layers[l]);if(observer)observeLayer(observer,l,h,r.layers[l],before,generated,n,start,count,pos,ids);}h=rmsNorm(device,h,weights.at(root+"norm.weight"),count,2560,1e-6f,true);if(last)final=slice(device,h,1,2560,(count-1)*2560,2560);else{if(!final.storage)final=device.floats(n*2560);P p{count,2560,0,0,start,0,0,0};device.dispatch("text_copy",{&h},{&final},&p,sizeof(p),(count*2560+127)/128);}device.drain();if(observer)observer({TextStage::FinalNorm,UINT32_MAX,h,count,2560,before,generated,n,start,count,r.layers[31].length,pos,ids});}r.tokens+=n;auto logits=linear(device,final,weights.at(root+"embed_tokens.weight"),last?1:n,true);device.drain();if(observer)observer({TextStage::Logits,UINT32_MAX,logits,last?1:n,248320,before,generated,n,last?n-1:0,last?1:n,r.tokens,pos,ids});return {final,logits,last?1:n};}catch(...){r.failed=true;for(auto& c:r.layers)c.failed=true;throw;}}
TextResult TextModel::prefill(TextRequest& r,const std::vector<uint32_t>& ids,const TextPositions& p,bool last,const TextObserver& observer){if(r.failed||r.retired)throw std::runtime_error("Failed/retired request cannot be reused");for(auto id:ids)if(id>=248320)throw std::runtime_error("Token outside pinned vocabulary");positions(p,uint32_t(ids.size()));auto h=embedding(device,weights.at(root+"embed_tokens.weight"),ids);return forward(r,h,p,last,observer,&ids);}
TextResult TextModel::advance(TextRequest& r,uint32_t id,const TextPositions& p,const TextObserver& observer){if(r.generated>=r.outputLimit||p.temporal.size()!=1)throw std::runtime_error("Retained output allowance exhausted");auto result=prefill(r,std::vector<uint32_t>{id},p,true,observer);r.generated++;return result;}
TextDecodeCohortResult TextModel::advanceCohort(const std::array<TextDecodeSlot,2>& slots,
 bool explicitOrderedB2,const std::array<TextObserver,2>& observers) {
 if(!explicitOrderedB2 || !experimental::gemmPadded32Enabled() ||
    std::strcmp(experimental::gemvB1Selection(),"ordered")!=0)
  throw std::invalid_argument("B2 decode requires explicit opt-in, padded32 and captured ordered B1; parallel32 excluded");
 const auto call=slots;
 const auto observe=observers;
 TextDecodeCohortResult result;
 std::array<uint32_t,2> mapping{},before{},generated{};
 std::array<std::vector<uint32_t>,2> ids;
 std::unordered_set<const TextRequest*> owners;
 std::unordered_set<const Storage*> backing;
 uint32_t count=0;
 // Include inactive owners in alias admission: advancing an active row cannot mutate a purportedly
 // inactive request through a shared backing buffer. Empty/retired inactive slots are allowed.
 for(uint32_t slot=0;slot<2;++slot) {
  const auto& s=call[slot];
  if(s.request) {
   if(!owners.insert(s.request).second)throw std::invalid_argument("Decode slots alias one request");
   for(const auto& c:s.request->layers)for(const Buffer* b:{&c.keys,&c.values,&c.conv,&c.recurrent})
    if(b->storage&&!backing.insert(b->storage.get()).second)
     throw std::invalid_argument("Decode slots/layers alias cache backing storage");
  }
  if(!s.active)continue;
  if(!s.request)throw std::invalid_argument("Active decode slot requires its own request");
  const auto& r=*s.request;
  positions(s.position,1);
  if(s.tokenId>=248320 || r.failed || r.retired || r.contextLimit!=16384 ||
     r.outputLimit!=12384 || r.generated>=r.outputLimit || r.tokens>=16384)
   throw std::invalid_argument("Active decode token/request/capacity invalid");
  for(uint32_t l=0;l<32;++l) {
   const auto& c=r.layers[l];
   if(c.failed||c.length!=r.tokens||(l%4==3?
      (!c.keys.storage||!c.values.storage||c.keys.words!=16384*1024||c.values.words!=16384*1024||
       c.keys.logicalElements!=uint64_t(16384)*1024||c.values.logicalElements!=uint64_t(16384)*1024||c.keys.packedBF16||c.values.packedBF16):
      (!c.conv.storage||!c.recurrent.storage||c.conv.words!=8192*4||c.recurrent.words!=32*128*128||
       c.conv.logicalElements!=8192*4||c.recurrent.logicalElements!=32*128*128||c.conv.packedBF16||c.recurrent.packedBF16)))
    throw std::invalid_argument("Active decode cache length/shape/storage invalid");
  }
  mapping[count++]=slot;before[slot]=r.tokens;generated[slot]=r.generated;ids[slot]={s.tokenId};result.active[slot]=true;
 }
 if(!count)throw std::invalid_argument("Decode cohort requires one or two active slots");
 try {
  if(count==1) {
   const auto slot=mapping[0];const auto& s=call[slot];
   result.slots[slot]=advance(*s.request,s.tokenId,s.position,observe[slot]);return result;
  }
  std::array<Buffer,2> hidden;
  for(uint32_t row=0;row<count;++row) {
   const auto slot=mapping[row];hidden[slot]=embedding(device,weights.at(root+"embed_tokens.weight"),ids[slot]);
  }
  // Gather/scatter are raw FP32 copies. Every submitted row belongs to exactly one stable slot.
  auto gather=[&](const std::array<Buffer,2>& values,uint32_t width) {
   Buffer packed=device.floats(count*width);
   for(uint32_t row=0;row<count;++row) {
    const auto slot=mapping[row];P p{1,width,0,0,row,0,0,0};
    device.dispatch("text_copy",{&values[slot]},{&packed},&p,sizeof(p),(width+127)/128);
   }
   return packed;
  };
  auto scatter=[&](const Buffer& packed,uint32_t width) {
   std::array<Buffer,2> values;
   for(uint32_t row=0;row<count;++row)values[mapping[row]]=slice(device,packed,1,width,row*width,width);
   device.drain();return values;
  };
  for(uint32_t l=0;l<32;++l) {
   const auto b=root+"layers."+std::to_string(l)+".";
   std::array<Buffer,2> residual;
   for(uint32_t row=0;row<count;++row) {
    const auto slot=mapping[row];const auto& s=call[slot];
    auto h=rmsNorm(device,hidden[slot],weights.at(b+"input_layernorm.weight"),1,2560,1e-6f,true);
    h=mix(l,h,1,s.position,s.request->layers[l]);
    residual[slot]=binary(device,hidden[slot],h,2560,0);
   }
   device.drain(); // Both independent state updates complete before sharing stateless work.
   auto packed=gather(residual,2560);
   auto norm=rmsNorm(device,packed,weights.at(b+"post_attention_layernorm.weight"),count,2560,1e-6f,true);
   auto gate=linear(device,norm,weights.at(b+"mlp.gate_proj.weight"),count);
   auto up=linear(device,norm,weights.at(b+"mlp.up_proj.weight"),count);
   auto act=unary(device,gate,count*9216,0);
   auto product=binary(device,act,up,count*9216,1);
   auto down=linear(device,product,weights.at(b+"mlp.down_proj.weight"),count);
   auto combined=binary(device,packed,down,count*2560,0);
   device.drain(); // Completed batched values before scattering or replacing per-slot hidden.
   hidden=scatter(combined,2560);
   for(uint32_t row=0;row<count;++row) {
    const auto slot=mapping[row];const auto& s=call[slot];
    if(observe[slot])observeLayer(observe[slot],l,hidden[slot],s.request->layers[l],before[slot],generated[slot],1,0,1,s.position,&ids[slot]);
   }
  }
  auto packed=gather(hidden,2560);
  auto norm=rmsNorm(device,packed,weights.at(root+"norm.weight"),count,2560,1e-6f,true);
  device.drain();
  auto normalized=scatter(norm,2560);
  for(uint32_t row=0;row<count;++row) {
   const auto slot=mapping[row];const auto& s=call[slot];
   if(observe[slot])observe[slot]({TextStage::FinalNorm,UINT32_MAX,normalized[slot],1,2560,before[slot],generated[slot],1,0,1,s.request->layers[31].length,s.position,&ids[slot]});
  }
  auto logits=linear(device,norm,weights.at(root+"embed_tokens.weight"),count,true);
  device.drain();
  auto separated=scatter(logits,248320);
  for(uint32_t row=0;row<count;++row)++call[mapping[row]].request->tokens;
  for(uint32_t row=0;row<count;++row) {
   const auto slot=mapping[row];const auto& s=call[slot];
   if(observe[slot])observe[slot]({TextStage::Logits,UINT32_MAX,separated[slot],1,248320,before[slot],generated[slot],1,0,1,s.request->tokens,s.position,&ids[slot]});
  }
  for(uint32_t row=0;row<count;++row) {
   const auto slot=mapping[row];++call[slot].request->generated;
   result.slots[slot]={normalized[slot],separated[slot],1};
  }
  return result;
 } catch(...) {
  // Device retains pending Storage references. Caller retains these poisoned caches and the
  // original exception; it must drain successfully (or retire the worker) before dropping them.
  for(uint32_t row=0;row<count;++row) {
   auto& r=*call[mapping[row]].request;r.failed=true;for(auto& c:r.layers)c.failed=true;
  }
  throw;
 }
}
}
